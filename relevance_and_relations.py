"""
relevance_and_relations.py

Adds two Groq-powered analysis steps on top of the existing pipeline output,
for every ENGLISH row in x_data_cleaned.csv:

  1. RELEVANCE CHECK - is the post GENUINELY discussing the Makkah/Mecca
     Joint Defence Pact or its direct fallout - not just mentioning one of
     the involved countries in an unrelated context? Written to a new
     "topic_relevant" column (True/False).

  2. RELATIONSHIP EXTRACTION - only for rows where topic_relevant is True
     AND actors_mentioned lists 2+ actors: what relationship is expressed
     BETWEEN EACH PAIR of those actors in that specific post (supportive /
     hostile / skeptical / neutral-reporting / mediating / unclear)?
     Appended to a new actor_relations.csv file (post_id, actor_1, actor_2,
     relation_type) - not written into x_data_cleaned.csv, since one post
     can produce multiple relation rows.

======================================================================
REWRITE HISTORY - READ THIS BEFORE TOUCHING THE PROMPTS/PACING BELOW
======================================================================

v1 (one post per API call) hit a WALL: after 6 hours it had only gotten
through 348/1499 rows, with retry waits compounding over time. Cause:
Groq's TOKENS-PER-MINUTE limit (8,000 TPM on the free tier for this
model), not the requests-per-minute limit v1 was pacing against. Two
things drove tokens per call up: (a) every single call resent the full
system prompt (100-150 tokens) for the sake of classifying ONE short
post - across ~2,400 calls that's ~300K tokens of pure repeated
overhead - and (b) multi-actor posts need one relation judgment per
PAIR of actors (a 5-actor post is 10 pairs), which blows up the
completion size for exactly the rows that also tend to have the longest
text.

v2 (this version) fixes this two ways:

  1. BATCHING: BATCH_SIZE_DEFAULT posts (default 7) are sent in ONE
     call, sharing ONE system prompt, and the model returns ONE JSON
     object with a "results" list keyed by post_id covering the whole
     batch - both the relevance verdict AND (where it applies) the pair
     relations for every post in that batch. This amortizes the fixed
     system-prompt overhead across many posts instead of paying it once
     per post, which is most of why v1's TPM usage was so much higher
     than the "obvious" per-post cost suggested. A batch is also cut
     short (fewer than BATCH_SIZE_DEFAULT posts) whenever its estimated
     input tokens would otherwise exceed MAX_BATCH_INPUT_TOKENS_DEFAULT -
     this dataset has some genuinely long posts (multi-paragraph, not
     just one-liners), and one of those shouldn't blow up a whole batch.

  2. TOKEN-AWARE PACING (TokenPacer below): v1 paced purely on a fixed
     seconds-between-calls number, which has no idea how many TOKENS
     each call actually used - fine for short posts, wrong for anything
     bigger. TokenPacer tracks a rolling 60-second window of ACTUAL
     token usage (Groq returns real prompt/completion/total token
     counts on every response - this isn't estimated after the fact,
     only the PRE-call sizing decision is an estimate) and makes a call
     wait if it would push the trailing-60s total over budget. It
     self-corrects: even if a pre-call size estimate is off, the very
     next call's wait decision is based on real numbers.

  Also: retry backoff is now hard-capped (RATE_LIMIT_BACKOFF_MAX_SEC)
  instead of growing unboundedly off Groq's Retry-After header, which is
  what let v1's waits "compound" the way you saw.

Net effect: total Groq calls for the whole dataset drop from ~2,100-2,400
(v1, one-to-two calls per row) to roughly 1499/BATCH_SIZE_DEFAULT ~ 215
calls (v2, one call per ~7 rows) - which also means the 1,000
requests/day free-tier cap (see below) is no longer the binding
constraint; a full run should fit in ONE day now, gated by the TPM pacer
instead.

v2.1 - a THIRD limit surfaced during real testing that neither v1 nor v2
accounted for: TOKENS PER DAY (TPD, 200,000/day for openai/gpt-oss-20b,
the model in use at the time), separate from tokens-per-minute. Groq's
rate-limit response HEADERS only expose remaining TPM and remaining RPD -
there's no header for remaining TPD, so TokenPacer (which only watches a
60-second window) had no way to see it coming. It only shows up in a 429
error's message text once it's already been hit. v1's inefficient
per-post calls, run for ~6 hours before the batching fix, had already
used 197,088 of that 200,000 daily budget by the time batching was
tested - so a very reasonable-looking --limit 20 smoke test failed
immediately. The fix (parse_rate_limit_hint below): read Groq's own
"Please try again in Xm Ys" text out of the error message and, if that
wait is longer than RATE_LIMIT_BACKOFF_MAX_SEC (it will be, for a TPD
wall), fail FAST with an accurate diagnosis instead of burning all 5
retries at 45s each first.

v2.2 - MODEL SWITCHED to qwen/qwen3.8-27b, away from openai/gpt-oss-20b,
specifically to get an INDEPENDENT daily token quota from the one v2.1
exhausted (each model on Groq gets its own separate RPM/RPD/TPM/TPD
bucket - confirmed straight off Groq's live rate-limits table, not
assumed). Two things ruled out before landing here:
  - llama-3.1-8b-instant (the original, pre-deprecation choice) - STILL
    dead. Re-confirmed live against Groq's deprecations page for this
    change: shut down August 16, 2026, does not appear anywhere in the
    current rate-limits table. Requesting it fails immediately with a
    404, same as before - there is no way to "switch back" to it.
  - qwen/qwen3-32b (the name that appears in this groq SDK version's own
    model type stub) - does NOT appear in Groq's LIVE rate-limits table
    at all, unlike qwen/qwen3.8-27b which does (checked twice, same
    result both times). An SDK type stub is a snapshot from whenever
    that SDK version was published, not a live source of truth - trusting
    it literally is exactly what went wrong with llama-3.1-8b-instant
    before, so it wasn't trusted blindly a second time.
  qwen/qwen3.8-27b's free-tier limits (RPM 30, RPD 1,000, TPM 8,000,
  TPD 2,000,000) are identical to gpt-oss-20b's on every axis EXCEPT
  TPD, where it's 10x bigger - directly addressing the wall that was
  actually hit, in the one place that mattered. It's also a reasoning
  model like gpt-oss-20b, but a strictly better-behaved one for this
  use case: it supports reasoning_effort="none" (gpt-oss-20b's minimum
  was "low" - it doesn't offer "none" at all), so reasoning tokens are
  no longer a factor here the way they were before - confirmed against
  Groq's reasoning-models doc, not assumed from the gpt-oss precedent.

v2.3 - a diagnostic spot-check of the "unclear" bucket (861 rows, ~23%
of all relations) after the full dataset finished (1499/1499, topic_relevant
split 1316 true / 183 false) found the label was mostly GENUINE ambiguity
(e.g. a post literally asking "how does Iran view the pact?" without
answering it) - but also found "unclear" was being used as a catch-all
for a different, more mechanical problem: the extraction prompt asked for
a relation_type for EVERY pair of actors listed for a post, even when the
post never actually said anything connecting that specific pair - it just
happened to mention both actors somewhere in a longer, multi-topic post.
Example: a post about India-Pakistan tensions possibly pushing India-Israel
cooperation closer together also happens to name the United States
elsewhere - the model had no real basis to relate Pakistan and the United
States to each other, but the prompt required an answer anyway, so it
defaulted to "unclear" instead of correctly saying "no relation to report
here."

FIX: FULL_BATCH_SYSTEM_PROMPT and RELATIONS_ONLY_BATCH_SYSTEM_PROMPT now
tell the model to first decide whether a pair is CONNECTED by the text at
all (some claim, characterization, or implication tying those two specific
actors together) before assigning a relation_type - if a pair is simply
co-mentioned with no connecting statement, the model is told to omit that
pair entirely rather than emit "unclear". "unclear" is now reserved for
pairs that ARE connected/implicated by the text but whose specific stance
can't be determined (the "how does Iran view the pact?" case). Both
prompts include worked SKIP/KEEP examples so this distinction is concrete,
not just described abstractly. No change was needed to parse_batch_response
or the data model - "omit the pair" was already representable (an empty or
shorter relations list), this is purely a prompt-level tightening.

Since this changes what the model outputs for a given post, it can only be
applied to POSTS RE-SENT under the new prompt - existing actor_relations.csv
rows extracted under the old, over-permissive prompt reflect the old
behavior and don't get retroactively fixed just by changing the prompt
text. Hence --recheck-relations (see RESUMABILITY below): it targets every
row already marked topic_relevant=True, purges that row's existing
actor_relations.csv entries (so old over-permissive rows don't linger next
to freshly re-extracted ones for the same post), resets relations_checked
back to NA for it, and re-runs RELATIONS EXTRACTION ONLY under the new
prompt - relevance (topic_relevant) is never touched or re-asked, since the
spot-check found no problem with that step and re-asking it would waste
tokens and risk flipping already-correct verdicts.

v2.3.1 - INCIDENT: the first --recheck-relations implementation (v2.3
above) marked its entire target set's relations_recheck_done=True AND
purged ALL of their actor_relations.csv rows UPFRONT, before a single
batch was confirmed - done deliberately, to make a RETRY of
--recheck-relations safe from re-purging already-finished rows. But when
the real run hit the qwen/qwen3.8-27b TPD wall at batch 48/203, this
meant all 1,316 targeted posts had already had their old relations
deleted, while only ~344 had gotten fresh ones - and relations_recheck_done
said "already handled" for all 1,316 regardless. A follow-up
--recheck-relations run then correctly found nothing matching its (wrong)
gating condition and reported "nothing left to recheck" - technically
accurate to what that column said, not to the real data. Audited directly
against the real files: 972 of 1,316 targeted posts had zero relation rows
at that point - not stale old-prompt data, just gone.

FIX: run_recheck_relations no longer reads relations_recheck_done for
target selection AT ALL - it selects purely on topic_relevant=True AND
relations_checked.isna(), the same column normal (non-recheck) resumability
has always used, now proven accurate (it's set exactly once per row, at
the exact moment that row's Groq result is confirmed - never upfront,
never in bulk). process_group gained two optional parameters to support
this without duplicating its batching/pacing/retry logic:
purge_before_batch (purges a batch's posts' stale relations only AFTER
that specific batch's result is parsed and confirmed error-free - never
before the call, never for a batch that fails to parse at all) and
recheck_marker_col (sets relations_recheck_done=True at the same per-row
confirmed moment as relations_checked - informational only now, never a
resume gate again). Net effect: an interruption at any point now leaves
AT MOST one batch's worth of posts (up to BATCH_SIZE_DEFAULT, i.e. 7) with
stale-but-not-yet-purged data, versus the entire remaining target set
before - and even that window only opens for posts whose fresh result
already arrived, closing the "purged with nothing to replace it" failure
mode entirely. The already-wrong relations_recheck_done values sitting on
disk (True for all 1,316, though only 344 are real) don't need manual
correction - they're simply never consulted again, and self-correct as
the outstanding 972 rows get reprocessed.

Also corrected in this pass: the earlier claim that qwen/qwen3.8-27b has
2,000,000 TPD was WRONG - re-verified directly against Groq's live
rate-limits table and it is 200,000 TPD, IDENTICAL to openai/gpt-oss-20b
on every axis (30 RPM / 1K RPD / 8K TPM / 200K TPD). The model switch in
v2.2 did get a SEPARATE quota bucket (Groq's per-model 429 error text and
a corroborating third-party source both indicate limits are tracked per
model, not pooled - though Groq's own docs don't explicitly confirm this
either way), just not a BIGGER one - that part of the v2.2 rationale was
incorrect and is retracted here.

v2.4 - a REAL 429 during the --recheck-relations run (after the v2.3.1
fix above) diagnosed a THIRD limit dimension neither TokenPacer nor the
public rate-limits table accounted for: a separate OUTPUT-tokens-per-minute
(OTPM) sub-limit of 1,000/min for qwen/qwen3.8-27b on this account - much
tighter than the 8,000 combined TPM limit already paced against, and
essentially invisible to that pacing since TPM counts prompt+completion
together while OTPM only counts completion tokens. A batch of
BATCH_SIZE_DEFAULT=7 posts, each with a real completion, could plausibly
produce close to or over 1,000 output tokens on its own even while
comfortably under the 8,000 TPM budget - explaining why the wall was being
hit almost every batch rather than occasionally. Re-checked Groq's public
rate-limits table for this change too: it does not list OTPM/ITPM figures
at all for any model - Groq's own docs explain these are account-specific
sub-limits, not part of the standard published table. The 1,000/min figure
here is taken directly from this account's real 429 error text, which is
the only place it's visible - it cannot be independently re-verified
against a public source the way the TPM/TPD numbers could be.

FIX: a second, independent TokenPacer instance (otpm_pacer) is now paced
against OTPM_LIMIT and fed ONLY output tokens (Groq's usage.completion_tokens),
in its own rolling 60-second window, separate from the existing pacer's
combined-token window. TokenPacer itself needed no changes - it was always
generic over "some token count in a 60s window"; call_groq_with_retry just
waits on both pacers (each independently loops until its own budget clears)
before every call, and records into both after. BATCH_SIZE_DEFAULT also
dropped from 7 to 4: a smaller batch means a smaller worst-case output per
call, which matters far more than call-count efficiency now that OTPM, not
TPM, is very likely the binding constraint for this dataset's typical
per-post output size. _LIMIT_TYPE_RE/_LIMIT_TYPE_NAMES also gained
OTPM/ITPM entries so a real OTPM 429 (if the pacer's estimate is ever off)
gets labeled correctly in the retry log instead of falling through to
"unknown limit".

v2.5 - a REAL request was rejected for asking for 1,040 max_tokens: Groq
enforces a hard PER-REQUEST ceiling of ~1,000 max_tokens, completely
separate from every pacing concept above (TPM/OTPM/TPD all govern how
much can be used OVER TIME; this instead caps how big any SINGLE request
is allowed to ask for, regardless of how well-paced it is). The v2.4 fix
correctly solved "hitting a wall almost every batch" (OTPM, a pacing
problem) but did nothing for this - a single well-paced request can still
be rejected outright if its own max_tokens is too high, and at
BATCH_SIZE_DEFAULT=4 with PER_POST_COMPLETION_CEILING_RELATIONS=300,
max_tokens for a full batch was 4*300=1,200 - over the ~1,000 hard cap on
its own, independent of pacing entirely.

FIX (request configuration, not pacing - deliberately no pacing changes in
this pass): MAX_TOKENS_PER_REQUEST_HARD_CAP=900 is now always included in
process_group's max_tokens calculation, so no request this script builds
can ever ask for more than 900, regardless of batch size or per-post
ceiling settings. BATCH_SIZE_DEFAULT also dropped again, 4 to 3: at 3
posts, 3*300=900 - exactly matching the hard cap with NO extra squeeze
needed, so relations-only requests (the only mode with outstanding work
left in this dataset right now) hit their natural, already-adequate
ceiling without the hard cap trimming any real budget off them. At
batch_size=4 the hard cap alone would have had to cut 1,200 down to 900,
which would have meant less headroom per post than the ceiling intended -
not a truncation this script has actually observed, but an avoidable risk
now removed by choosing a batch size whose natural ceiling already fits.

======================================================================

RESUMABILITY (unchanged since v1, still the same contract): a row is
"done" once topic_relevant is set AND relations_checked is True.
relations_checked=True means "nothing more owed here" - either a
relation-extraction result was obtained, or the row didn't qualify (not
relevant, or fewer than 2 actors). Both columns start blank/NA, so
re-running this script only re-does work that's still outstanding -
already-processed rows are NOT re-sent. This still works exactly the
same under batching AND under sampling (see --sample-size below); only
how a "todo" row's task gets scheduled changed, not what counts as
"todo".

RECHECK-RELATIONS (v2.3, fixed in v2.3.1 - see that section above for the
incident): --recheck-relations re-runs relation extraction (only - never
relevance) under the tightened prompt for every row with topic_relevant=True
whose relations_checked is not yet True. relations_checked is the ONLY
signal this trusts for what's outstanding - the same column normal
resumability has always used. A third column, relations_recheck_done, is
still written for auditability ("was this post's current relation data
produced under the tightened prompt") but is NEVER read back as a gate -
that's what v2.3.1 fixed. A post's stale actor_relations.csv rows are
purged only once THAT post's fresh result is confirmed (see
purge_before_batch in process_group), not upfront for the whole target
set, so an interruption at any point never deletes more than one batch's
worth of posts' data without a replacement ready. --recheck-relations is
safe to interrupt and resume by simply running it again (or a plain run
with no flags) - relations_checked accurately reflects exactly what's left
either way. See --recheck-relations's --help text for the exact behavior.

SAMPLING (v2.2, new): --sample-size caps how many English rows this
script tries to have "done" in total, prioritizing the highest-engagement
(likes + retweets) rows among what's still outstanding. Rows already
done are NEVER discarded or re-processed just because they fall outside
the sample - "target count" means "how many should be done after this
run", not "how many to touch this run". See build_sample_idx and its
call site in main() for exactly how this interacts with resumability.

MODEL: qwen/qwen3.8-27b (see v2.2 above for why). If Groq deprecates or
renames this one too, the MODEL constant below is the only thing that
needs to change - but re-verify against Groq's LIVE rate-limits table
and deprecations page first, the way this change did, rather than
trusting any single source (including this file's own comments, which
are a snapshot from whenever they were last verified).

RATE LIMITS (Groq free tier, qwen/qwen3.8-27b - see v2.3.1/v2.4/v2.5 above
for the correction history on these numbers, all re-verified against
Groq's live rate-limits table or this account's own real errors): 30
requests/minute, 8,000 tokens/minute (TPM), 200,000 tokens/day (TPD),
1,000 requests/day (RPD) - IDENTICAL to openai/gpt-oss-20b on every one of
these axes, from the public table. PLUS a separate, account-specific,
NOT-publicly-documented output-tokens-per-minute sub-limit (OTPM) of
1,000/min (see v2.4) - paced against via otpm_pacer. PLUS a hard
PER-REQUEST max_tokens ceiling of ~1,000 (see v2.5) - NOT a pacing concept
at all, just a cap on how big any one request's max_tokens may be, enforced
via MAX_TOKENS_PER_REQUEST_HARD_CAP regardless of pacing. TPD/RPD are not
expected to be realistic concerns for this dataset's remaining backlog at
the current batch size - but the resumable checkpointing means even if any
of these DID bind, nothing would be lost.

GOTCHA CARRIED FORWARD, but IMPROVED: qwen3.8-27b is also a REASONING
model, like gpt-oss-20b was - it can spend completion-token budget on
internal reasoning before the visible answer. Unlike gpt-oss-20b though,
it supports reasoning_effort="none", which is used here to turn that
overhead off entirely rather than just minimizing it. reasoning_format=
"parsed" is still set too (keeps any reasoning that does occur out of
message.content), and per-batch max_completion_tokens is still sized
generously as a ceiling, not a target, as a safety margin.

INCREMENTAL SAVES: x_data_cleaned.csv is checkpointed after every batch
(atomic write via a .tmp file + os.replace, so a crash mid-write can't
corrupt the real file), and actor_relations.csv rows are appended as
they're produced rather than held in memory - a multi-hour run that dies
partway through should not lose completed work.

Requires GROQ_API_KEY in .env (loaded via python-dotenv, same convention
as the rest of this project).

This script makes real Groq API calls and cannot be exercised end-to-end
from the environment that wrote it (no network path to api.groq.com from
there) - see test_relevance_and_relations_offline.py for the offline
mocked-batch-response test that verifies the batching/parsing logic
instead. Run this file for real yourself:

    python relevance_and_relations.py                       # run until "done" count reaches --sample-size
    python relevance_and_relations.py --sample-size 0        # no cap - process every remaining row
    python relevance_and_relations.py --sample-size 900       # raise the target (e.g. once already past 400)
    python relevance_and_relations.py --limit 20              # smoke test: first 20 unprocessed rows only
    python relevance_and_relations.py --batch-size 2          # even smaller batches than the v2.5 default of 3
    python relevance_and_relations.py --max-batch-input-tokens 1500   # cut batches shorter for very long posts
    python relevance_and_relations.py --recheck-relations     # v2.3: re-extract relations (not relevance)
                                                                # for every topic_relevant=True row, under
                                                                # the tightened prompt - see v2.3 docstring
    python relevance_and_relations.py --recheck-relations --limit 20  # smoke test the recheck on 20 rows first
"""

import argparse
import ast
import csv
import json
import os
import re
import sys
import time
from collections import deque

import pandas as pd
from dotenv import load_dotenv

import groq
from groq import Groq

# Preventive fix, applied pipeline-wide after a real incident: two other
# steps in this pipeline (preprocess_x_text.py, build_network_v2.py) each
# crashed with UnicodeEncodeError while printing scraped X/Twitter text
# (an emoji) to a Windows console defaulting to a single-byte codepage
# (cp1252) that can't represent it. This script's own warning/checkpoint
# prints can include text sourced from Groq's responses (error reasons,
# parse-failure snippets) about the same scraped posts, and it runs on the
# same unattended schedule - reconfiguring stdout/stderr to UTF-8 with
# errors="replace" here too closes off the same crash class pre-emptively -
# an unprintable character is swapped for a placeholder instead of ever
# being able to crash this step.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
X_DATA_CSV = os.path.join(DATA_DIR, "x_data_cleaned.csv")
RELATIONS_CSV = os.path.join(DATA_DIR, "actor_relations.csv")
FAILURE_LOG = os.path.join(DATA_DIR, "relation_extraction_failures.log")

MODEL = "qwen/qwen3.8-27b"

RELATION_TYPES = ["supportive", "hostile", "skeptical", "neutral-reporting", "mediating", "unclear"]

# --- batching ---
# v2.5: reduced again, 4 to 3 - see the v2.5 docstring section. A REAL request was
# rejected for asking for 1,040 max_tokens: Groq enforces a hard ~1,000-per-REQUEST
# ceiling that is completely separate from (and not solved by) OTPM pacing - pacing
# controls WHEN a call goes out, not how big any single call is allowed to be. At
# batch_size=3, PER_POST_COMPLETION_CEILING_RELATIONS(300) * 3 = 900 exactly - matching
# MAX_TOKENS_PER_REQUEST_HARD_CAP below with NO extra squeeze needed, so relations-only
# requests (the only mode with any outstanding work left in this dataset) are naturally
# under the hard cap without truncating legitimate output. batch_size=4 would need
# 1200 - > 900 forced down to 900, shaving real budget off every request.
BATCH_SIZE_DEFAULT = 3                    # max posts per API call
MAX_BATCH_INPUT_TOKENS_DEFAULT = 2000     # also cut a batch short if estimated input tokens would exceed this
PER_POST_COMPLETION_CEILING_FULL = 450    # per post, relevance + (if applicable) relations, in "full" mode
PER_POST_COMPLETION_CEILING_RELATIONS = 300  # per post, relations only (relevance already known)
ABSOLUTE_MAX_COMPLETION_TOKENS = 4000     # pre-v2.5 ceiling concept - kept for history/reference, but
                                            # MAX_TOKENS_PER_REQUEST_HARD_CAP below is always the tighter,
                                            # binding one in practice now (900 < 4000)
# v2.5, new: Groq's REAL hard per-request max_tokens ceiling is ~1,000 - confirmed via an
# actual rejected request that asked for 1,040. This is NOT a pacing/rate-limit concept at
# all (OTPM/TPM govern how much you can use over TIME; this governs the size of any ONE
# request, full stop) - so no amount of pacing fixes it; the fix is capping what's ever
# requested. 900 leaves a 100-token safety margin below Groq's ~1,000 ceiling.
MAX_TOKENS_PER_REQUEST_HARD_CAP = 900

# --- sampling ---
SAMPLE_SIZE_DEFAULT = 400  # target TOTAL "done" rows (already-done + newly-processed), not new rows alone;
                             # 0 (or --sample-size 0) disables sampling and processes every remaining row

# --- token-aware pacing ---
TPM_LIMIT = 8000          # Groq free tier, qwen/qwen3.8-27b (see docstring) - same TPM as gpt-oss-20b had
TPM_SAFETY_FACTOR = 0.85  # target 85% of the real cap, leaving headroom for estimate error

# --- output-token-per-minute pacing (v2.4, new - see the v2.4 docstring section) ---
# Confirmed via a REAL 429 on this account: qwen/qwen3.8-27b is subject to a separate,
# much tighter OUTPUT-tokens-per-minute (OTPM) sub-limit of 1,000/min, ON TOP OF (not
# instead of) the combined 8,000 TPM limit TokenPacer above already paces against. This
# number is NOT in Groq's public rate-limits table - re-checked that table directly for
# this change, and it only documents the combined TPM figure. Groq's own docs explain
# why: "some organizations are also subject to separate per-minute limits on input
# tokens (ITPM) and output tokens (OTPM)" - account-specific, visible only on that
# account's own Limits page or (as happened here) in a 429's error text once hit. So
# 1,000 is taken from this account's real error message, not a public source - flagging
# that plainly since it can't be independently re-verified the way TPM/TPD could be.
OTPM_LIMIT = 1000
OTPM_SAFETY_FACTOR = 0.85  # same margin as TPM_SAFETY_FACTOR, for the same reason

# --- retry/backoff ---
MAX_RETRIES = 5
RETRY_BASE_DELAY_SEC = 3          # backoff for connection/5xx errors (attempt * this, capped below)
RATE_LIMIT_BACKOFF_SEC = 15       # starting backoff after a 429, before retrying
RATE_LIMIT_BACKOFF_MAX_SEC = 45   # HARD CAP - a single retry never waits longer than this, no matter
                                    # what Retry-After says or how many attempts have failed. v1's
                                    # waits grew unboundedly off Retry-After during sustained TPM
                                    # throttling; that's what "compounding over time" was.


# --------------------------------------------------------------------------
# Prompts (batch versions - each call covers multiple posts at once)
# --------------------------------------------------------------------------

# v2.3: the "when to connect a pair at all" instruction + worked examples are shared verbatim
# between both prompts (full and relations-only) so a row processed under either mode gets the
# identical extraction standard - see the v2.3 docstring section for why this changed.
_PAIR_CONNECTION_INSTRUCTION = (
    "for EVERY pair of those actors, first decide whether the post's text actually CONNECTS that "
    "specific pair - some claim, characterization, or implication tying those two actors together "
    "(support, opposition, criticism/skepticism, mediation, or even a plain factual link, like being "
    "co-signatories or named together in one development). If the pair is never connected - the post "
    "just happens to mention both actors somewhere, in a longer or multi-topic post, with no "
    "statement relating them to EACH OTHER - DO NOT output a relation for that pair at all; leave it "
    "out of the relations list entirely. Do NOT use 'unclear' as a catch-all for pairs that simply "
    "are not connected in the text: 'unclear' is ONLY for a pair that IS connected/implicated by the "
    "text but whose specific stance (supportive vs hostile vs skeptical, etc.) genuinely can't be "
    "determined. For every pair that IS connected, choose exactly one relation_type from this fixed "
    "set: " + ", ".join(RELATION_TYPES) + ".\n\n"
    "Examples:\n"
    "- SKIP the pair (do not emit 'unclear' either): a post mainly about India-Pakistan tensions "
    "notes this 'may push India and Israel to deepen cooperation' while separately mentioning "
    "Pakistan's role in the pact elsewhere in the post. If the post never makes a statement "
    "connecting Pakistan and the United States to EACH OTHER, do not emit a Pakistan/United States "
    "relation at all, even though both names appear somewhere in the post.\n"
    "- KEEP, relation_type='unclear': a post asks 'How does Iran view the Saudi-Pakistan-Turkey "
    "Mecca pact?' as an open question, without answering it. Iran and Saudi Arabia ARE connected by "
    "the text (the question itself relates them to the pact and to each other's position on it), but "
    "the actual stance is genuinely left unresolved.\n"
    "- KEEP, judgment call but real - do not default this to 'unclear' either: an article notes "
    "'a more self-reliant regional bloc... could complicate U.S. plans and alliances' while "
    "discussing Turkiye's role in the pact. This DOES connect Turkiye and the United States "
    "(implying friction/concern from the U.S. side), even though it is phrased as measured reporting "
    "rather than an explicit stance - output the best-fitting relation_type (e.g. 'neutral-reporting' "
    "or 'skeptical') rather than omitting it or defaulting to 'unclear'.\n\n"
)

FULL_BATCH_SYSTEM_PROMPT = (
    "You are analyzing a batch of social media posts for a research project studying public "
    "reaction to the Makkah/Mecca Joint Defence Pact, a mutual-defense agreement signed August "
    "7, 2026 between Saudi Arabia, Turkiye, and Pakistan. You will be given several posts, each "
    "labeled with a post_id and a list of actors (countries) it mentions.\n\n"
    "For EVERY post, independently:\n"
    "1. topic_relevant: true if the post is GENUINELY discussing the pact itself or its direct "
    "fallout/consequences - not just mentioning one of the involved countries in an unrelated "
    "context. Otherwise false.\n"
    "2. relations: ONLY if topic_relevant is true AND 2 or more actors are listed for that post: "
    + _PAIR_CONNECTION_INSTRUCTION +
    "If topic_relevant is false, or fewer than 2 actors are listed for that post, relations must be "
    "an empty list.\n\n"
    "Reply with ONLY a JSON object of exactly this shape, no other text, no markdown fences, "
    "covering EVERY post_id given to you exactly once:\n"
    '{"results": [{"post_id": "<id>", "topic_relevant": true/false, '
    '"relations": [{"actor_1": "<name>", "actor_2": "<name>", "relation_type": "<type>"}]}]}'
)

RELATIONS_ONLY_BATCH_SYSTEM_PROMPT = (
    "You are analyzing a batch of social media posts about the Makkah/Mecca Joint Defence Pact "
    "(signed August 7, 2026 between Saudi Arabia, Turkiye, and Pakistan) for a research project. "
    "Every post below has ALREADY been confirmed as genuinely discussing the pact, and each is "
    "labeled with a post_id and the actors (countries) it mentions.\n\n"
    "For EVERY post, " + _PAIR_CONNECTION_INSTRUCTION +
    "Reply with ONLY a JSON object of exactly this shape, no other text, no markdown fences, "
    "covering EVERY post_id given to you exactly once:\n"
    '{"results": [{"post_id": "<id>", "relations": [{"actor_1": "<name>", "actor_2": "<name>", '
    '"relation_type": "<type>"}]}]}'
)


def build_batch_prompt_body(batch_rows):
    """batch_rows: list of {"post_id", "text", "actors"} dicts."""
    lines = []
    for i, row in enumerate(batch_rows, 1):
        actors_str = ", ".join(row["actors"]) if row["actors"] else "(none detected)"
        lines.append(f"--- Post {i} (post_id={row['post_id']}, actors={actors_str}) ---\n{row['text']}")
    return "\n\n".join(lines)


def build_full_batch_messages(batch_rows):
    return [
        {"role": "system", "content": FULL_BATCH_SYSTEM_PROMPT},
        {"role": "user", "content": build_batch_prompt_body(batch_rows)},
    ]


def build_relations_only_batch_messages(batch_rows):
    return [
        {"role": "system", "content": RELATIONS_ONLY_BATCH_SYSTEM_PROMPT},
        {"role": "user", "content": build_batch_prompt_body(batch_rows)},
    ]


# --------------------------------------------------------------------------
# Response parsing (pure functions - this is what the offline test exercises)
# --------------------------------------------------------------------------

def _extract_json_block(raw):
    """Best-effort fallback: pull the first {...} block out of a reply that
    wrapped its JSON in markdown fences or added stray commentary."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    return match.group(0) if match else None


def parse_batch_response(raw, expected, mode):
    """
    expected: dict of post_id -> {"actors": [...]} for every post that was
    sent in this batch (mode="full") or that's already confirmed relevant
    (mode="relations_only").

    Returns (results, batch_error).

    batch_error is set (and results is {}) ONLY when the whole reply
    couldn't be parsed as JSON at all - in that case every post in the
    batch is the caller's problem to retry next run.

    On success, results is a dict post_id -> {"topic_relevant", "relations",
    "error"} for EVERY post_id in `expected` - including ones the model
    dropped from its reply ("missing from response") or answered badly
    ("topic_relevant missing or not boolean", etc). `error` is None only
    for a post that came back clean. relations is always a list of
    (actor_1, actor_2, relation_type) tuples (possibly empty), with any
    pair naming an actor outside that post's actual actor list silently
    dropped (hallucination guard) - same defensive behavior as v1.
    """
    if raw is None:
        return {}, "empty response"

    parsed = None
    for candidate in (raw, _extract_json_block(raw)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue
    if parsed is None:
        return {}, "response was not valid JSON"

    items = parsed.get("results") if isinstance(parsed, dict) else None
    if items is None or not isinstance(items, list):
        return {}, "JSON was missing a 'results' list"

    out = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        post_id = str(item.get("post_id", "")).strip()
        if post_id not in expected or post_id in out:
            continue  # unknown/hallucinated post_id, or a duplicate - ignore
        actors = expected[post_id]["actors"]
        entry = {"error": None}

        if mode == "full":
            tr = item.get("topic_relevant")
            if not isinstance(tr, bool):
                out[post_id] = {"topic_relevant": None, "relations": None,
                                 "error": "topic_relevant missing or not boolean"}
                continue
            entry["topic_relevant"] = tr
        else:
            entry["topic_relevant"] = True  # already known True in relations_only mode

        relations_expected = entry["topic_relevant"] is True and len(actors) >= 2
        relations_raw = item.get("relations", [])
        if isinstance(relations_raw, list):
            actor_set = set(actors)
            relations = []
            for r in relations_raw:
                if not isinstance(r, dict):
                    continue
                a1 = str(r.get("actor_1", "")).strip()
                a2 = str(r.get("actor_2", "")).strip()
                rtype = str(r.get("relation_type", "")).strip().lower()
                if not a1 or not a2 or a1 == a2:
                    continue
                if a1 not in actor_set or a2 not in actor_set:
                    continue  # hallucinated actor not actually in this post - drop it
                if rtype not in RELATION_TYPES:
                    rtype = "unclear"
                relations.append((a1, a2, rtype))
            entry["relations"] = relations
        else:
            entry["relations"] = []
            if relations_expected:
                entry["error"] = "relations field was not a list where relations were expected"

        out[post_id] = entry

    for post_id in expected:
        if post_id not in out:
            out[post_id] = {"topic_relevant": None, "relations": None, "error": "missing from response"}

    return out, None


def parse_actors(raw):
    """actors_mentioned is stored as str(sorted(list)), e.g. "['Iran', 'Pakistan']"."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)) or not str(raw).strip():
        return []
    try:
        val = ast.literal_eval(str(raw))
        if isinstance(val, list):
            return [str(a) for a in val]
    except (ValueError, SyntaxError):
        pass
    return []


def estimate_tokens(text):
    """Rough chars/4 heuristic - good enough for pre-call sizing decisions.
    Post-call decisions use Groq's own real usage numbers instead (see
    TokenPacer.record)."""
    return max(1, len(text) // 4)


def build_batches(idx_list, df, max_rows_per_batch, max_input_tokens_per_batch):
    """Greedily groups row indices into batches, cutting a batch short (even
    under max_rows_per_batch) once its estimated input tokens would exceed
    max_input_tokens_per_batch - this dataset has some posts that are long
    multi-paragraph essays, not just one-liners, and one of those shouldn't
    blow up a whole batch's token budget by itself."""
    batches = []
    current, current_tokens = [], 0
    for idx in idx_list:
        text = str(df.at[idx, "clean_text"])
        t = estimate_tokens(text)
        if current and (len(current) >= max_rows_per_batch or current_tokens + t > max_input_tokens_per_batch):
            batches.append(current)
            current, current_tokens = [], 0
        current.append(idx)
        current_tokens += t
    if current:
        batches.append(current)
    return batches


def build_sample_idx(df, en_mask, sample_size):
    """Returns (in_scope_idx, already_done_count, newly_sampled_count).

    in_scope_idx is a set of df.index values: every already-done English
    row (relations_checked == True - see the resumability contract in the
    module docstring) PLUS, if sample_size is set, the top
    (sample_size - already_done_count) still-outstanding English rows by
    (likes + retweets) engagement, descending. Already-done rows are ALWAYS
    included regardless of their engagement - sampling only decides which
    NOT-yet-done rows get worked on this run, never discards finished work.

    sample_size <= 0 (or None) disables sampling: every remaining row is
    in scope, matching this script's original (pre-sampling) behavior."""
    # relations_checked is a nullable "boolean" column - NA (not-yet-done)
    # compared with `== True` yields NA, not False (pandas' Kleene logic),
    # which would silently make every not-done row vanish from both
    # done_mask AND its negation instead of correctly landing in "not done".
    # .fillna(False) first sidesteps that - same pattern already used for
    # needs_relations in main().
    done_mask = en_mask & df["relations_checked"].fillna(False).astype(bool)
    done_idx = set(df.index[done_mask])

    remaining_mask = en_mask & ~done_mask
    if not sample_size or sample_size <= 0:
        remaining_idx = df.index[remaining_mask]
        return done_idx | set(remaining_idx), len(done_idx), len(remaining_idx)

    additional_needed = max(0, sample_size - len(done_idx))
    if additional_needed == 0:
        return done_idx, len(done_idx), 0

    likes = pd.to_numeric(df.loc[remaining_mask, "likes"], errors="coerce").fillna(0)
    retweets = pd.to_numeric(df.loc[remaining_mask, "retweets"], errors="coerce").fillna(0)
    engagement = (likes + retweets).sort_values(ascending=False)
    top_idx = engagement.head(additional_needed).index
    return done_idx | set(top_idx), len(done_idx), len(top_idx)


# --------------------------------------------------------------------------
# Token-aware pacing
# --------------------------------------------------------------------------

class TokenPacer:
    """Tracks a rolling 60-second window of token usage and makes a call
    wait if it would push the trailing total over budget. `clock` and
    `sleep_fn` are injectable so the offline test can exercise the
    accounting logic without real sleeping."""

    def __init__(self, tpm_limit, safety_factor=TPM_SAFETY_FACTOR, clock=time.monotonic, sleep_fn=time.sleep):
        self.budget = tpm_limit * safety_factor
        self.window = deque()  # (timestamp, tokens)
        self.clock = clock
        self.sleep_fn = sleep_fn

    def _prune(self, now):
        cutoff = now - 60
        while self.window and self.window[0][0] < cutoff:
            self.window.popleft()

    def used_last_minute(self):
        self._prune(self.clock())
        return sum(t for _, t in self.window)

    def wait_for_budget(self, estimated_tokens, label=""):
        waited = False
        while True:
            now = self.clock()
            self._prune(now)
            used = sum(t for _, t in self.window)
            if used + estimated_tokens <= self.budget:
                return waited
            waited = True
            sleep_for = 5.0
            if self.window:
                sleep_for = max(2.0, min(15.0, 60 - (now - self.window[0][0])))
            print(
                f"  [token-pacer]{' ' + label if label else ''} ~{used:.0f}/{self.budget:.0f} tokens "
                f"used in the last 60s - waiting ~{sleep_for:.0f}s before a ~{estimated_tokens}-token call"
            )
            self.sleep_fn(sleep_for)

    def record(self, tokens):
        self.window.append((self.clock(), tokens))


# --------------------------------------------------------------------------
# Parsing Groq's own 429 error message
# --------------------------------------------------------------------------
#
# Groq's rate-limit RESPONSE HEADERS only expose two of the four limits that
# actually apply (x-ratelimit-*-tokens is TPM only; x-ratelimit-*-requests is
# RPD only - confirmed straight from Groq's docs). There is no header for
# remaining TOKENS PER DAY, which is exactly the one that bit this script:
# TPM pacing (TokenPacer above) can be perfectly healthy while TPD is
# nearly exhausted, and the only place that shows up at all is inside the
# 429 error MESSAGE TEXT once it's already happened, e.g.:
#   "...on tokens per day (TPD): Limit 200000, Used 197088, Requested 4151.
#    Please try again in 8m55.248s..."
# So this parses that text as a best-effort signal for two things: which of
# the four limit types was actually hit, and how long Groq itself says to
# wait - which matters a lot, because a "try again in 8m55s" wall will never
# clear from a few 45-second-capped retries, and retrying into it 5 times
# anyway just wastes minutes before failing with a generic, unhelpful message
# (this is literally what happened during testing: 5 x 45s = 225s spent
# retrying a wall that needed ~535s, then a hint that guessed the WRONG
# limit type). Below, a hinted wait longer than RATE_LIMIT_BACKOFF_MAX_SEC
# skips retrying altogether and fails fast with an accurate diagnosis.

# v2.4: OTPM/ITPM added - see the v2.4 docstring section. These are account-specific
# sub-limits Groq's public rate-limits table doesn't list at all, so this regex is the
# only place that recognizes them - without it, a real OTPM/ITPM 429 would have fallen
# through to "unknown limit" in the retry log instead of being correctly labeled.
_LIMIT_TYPE_RE = re.compile(r"\((TPD|TPM|OTPM|ITPM|RPD|RPM)\)")
_WAIT_RE = re.compile(r"try again in\s+(?:([\d.]+)h)?(?:([\d.]+)m)?(?:([\d.]+)s)?", re.IGNORECASE)

_LIMIT_TYPE_NAMES = {
    "TPD": "daily TOKEN (TPD)", "TPM": "per-minute TOKEN (TPM)",
    "OTPM": "per-minute OUTPUT TOKEN (OTPM)", "ITPM": "per-minute INPUT TOKEN (ITPM)",
    "RPD": "daily REQUEST (RPD)", "RPM": "per-minute REQUEST (RPM)",
}


def parse_rate_limit_hint(message):
    """Returns (limit_type_or_None, wait_seconds_or_None) parsed from a
    Groq 429 error message's text. Either or both can come back None if the
    message doesn't match the expected shape - callers must handle that."""
    limit_type = None
    m = _LIMIT_TYPE_RE.search(message)
    if m:
        limit_type = m.group(1)
    wait_seconds = None
    m2 = _WAIT_RE.search(message)
    if m2 and any(m2.groups()):
        h, mnt, s = (float(g) if g else 0.0 for g in m2.groups())
        wait_seconds = h * 3600 + mnt * 60 + s
    return limit_type, wait_seconds


# --------------------------------------------------------------------------
# Groq call wrapper with token-aware pacing + capped retry/backoff
# --------------------------------------------------------------------------

def call_groq_with_retry(client, messages, max_completion_tokens, label, pacer, estimated_tokens,
                          response_format=None, otpm_pacer=None, estimated_output_tokens=None):
    """pacer/estimated_tokens: existing combined-TPM pacing (unchanged).
    otpm_pacer/estimated_output_tokens (v2.4, optional): a SECOND, independent
    TokenPacer instance paced against OTPM_LIMIT instead of TPM_LIMIT, tracking
    only OUTPUT tokens (Groq's usage.completion_tokens) in its own rolling 60s
    window. TokenPacer itself needed no changes to support this - it was
    already generic over "some kind of token count in a 60s window"; this
    just instantiates a second one with a different budget and feeds it a
    different number. Both pacers must clear before a call proceeds - see
    the v2.4 docstring section for why a single combined-TPM pacer wasn't
    enough (OTPM is a separate, tighter sub-limit on the same account, not
    covered by the TPM check at all)."""
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        pacer.wait_for_budget(estimated_tokens, label=label)
        if otpm_pacer is not None:
            otpm_pacer.wait_for_budget(estimated_output_tokens or 0, label=f"{label} [OTPM]")
        try:
            kwargs = dict(
                model=MODEL,
                messages=messages,
                temperature=0,
                max_completion_tokens=max_completion_tokens,
                # qwen3.8-27b is a reasoning model - see module docstring. Unlike
                # gpt-oss-20b (the previous model), it supports "none", which is
                # used here to turn reasoning-token overhead off entirely.
                reasoning_effort="none",
                reasoning_format="parsed",
            )
            if response_format:
                kwargs["response_format"] = response_format
            resp = client.chat.completions.create(**kwargs)

            usage = getattr(resp, "usage", None)
            actual_tokens = getattr(usage, "total_tokens", None) if usage else None
            pacer.record(actual_tokens if actual_tokens else estimated_tokens)
            if otpm_pacer is not None:
                actual_output_tokens = getattr(usage, "completion_tokens", None) if usage else None
                otpm_pacer.record(actual_output_tokens if actual_output_tokens else (estimated_output_tokens or 0))

            message = resp.choices[0].message
            content = message.content
            if not content:
                finish_reason = resp.choices[0].finish_reason
                reasoning_chars = len(message.reasoning) if getattr(message, "reasoning", None) else 0
                print(
                    f"  [{label}] got empty content back (finish_reason={finish_reason}, "
                    f"{reasoning_chars} reasoning char(s), used {actual_tokens or 'unknown'} tokens) - "
                    f"likely ran out of max_completion_tokens before finishing."
                )
            return content
        except groq.AuthenticationError as e:
            raise RuntimeError(
                "Groq rejected the API key (401 Unauthorized). Check GROQ_API_KEY in .env."
            ) from e
        except groq.BadRequestError as e:
            raise RuntimeError(f"[{label}] Groq rejected the request (400): {e}") from e
        except groq.NotFoundError as e:
            # Retrying can't fix a bad model ID - fail immediately instead of
            # burning MAX_RETRIES on something that will never succeed.
            raise RuntimeError(
                f"[{label}] Groq says model '{MODEL}' doesn't exist or isn't accessible (404): {e}\n"
                f"Groq may have deprecated it - check https://console.groq.com/docs/deprecations "
                f"and update the MODEL constant near the top of this file."
            ) from e
        except groq.RateLimitError as e:
            msg = str(e)
            limit_type, hinted_wait = parse_rate_limit_hint(msg)

            wait = hinted_wait
            if wait is None:
                resp_obj = getattr(e, "response", None)
                retry_after = resp_obj.headers.get("retry-after") if resp_obj is not None else None
                if retry_after:
                    try:
                        wait = float(retry_after) + 1
                    except ValueError:
                        wait = None
            if wait is None:
                wait = RATE_LIMIT_BACKOFF_SEC

            if wait > RATE_LIMIT_BACKOFF_MAX_SEC:
                # A wait this long won't clear from a few capped retries - this
                # is almost certainly a DAILY limit (TPD or RPD), not an
                # ordinary transient per-minute blip. Fail fast with an
                # accurate diagnosis instead of burning MAX_RETRIES uselessly.
                limit_name = _LIMIT_TYPE_NAMES.get(limit_type, "rate")
                raise RuntimeError(
                    f"[{label}] Hit Groq's {limit_name} limit. Groq says: {msg}\n"
                    f"Groq itself estimates ~{wait/60:.1f} min until this clears. That's a ROLLING "
                    f"window (usage 'ages out' ~24h after it happened, not a fixed once-a-day reset "
                    f"at midnight), so it should free up faster than 'wait until tomorrow' - but "
                    f"retrying every {RATE_LIMIT_BACKOFF_MAX_SEC}s right now won't help, so this "
                    f"script isn't going to try. Just re-run this exact command again once you've "
                    f"waited - progress already made is saved, nothing is lost. If this is TPD "
                    f"(daily tokens) and you're still close to the ceiling shortly after waiting, "
                    f"that likely means an earlier, less efficient run used most of today's budget - "
                    f"give it more time (a few hours, or the next day) rather than retrying repeatedly."
                ) from e

            wait = min(wait, RATE_LIMIT_BACKOFF_MAX_SEC)  # hard cap - see constant's comment
            print(f"  [{label}] attempt {attempt}/{MAX_RETRIES}: 429 rate-limited ({limit_type or 'unknown'} limit) - waiting {wait:.0f}s")
            time.sleep(wait)
            last_err = e
        except (groq.APIConnectionError, groq.InternalServerError, groq.APITimeoutError) as e:
            wait = min(RETRY_BASE_DELAY_SEC * attempt, RATE_LIMIT_BACKOFF_MAX_SEC)
            print(f"  [{label}] attempt {attempt}/{MAX_RETRIES}: {type(e).__name__} - waiting {wait:.0f}s")
            time.sleep(wait)
            last_err = e
    hint = ""
    if isinstance(last_err, groq.RateLimitError):
        hint = (
            "\nThis is a RateLimitError that didn't clear after several short retries (each capped "
            "at 45s) - with OTPM-aware pacing this should be rare (see v2.4 in the module docstring). "
            "Try --batch-size 2 to shrink calls further, or wait longer and re-run. "
            "Progress already made is saved either way."
        )
    raise RuntimeError(f"[{label}] gave up after {MAX_RETRIES} attempts: {last_err}{hint}")


# --------------------------------------------------------------------------
# Data I/O
# --------------------------------------------------------------------------

def require_columns(df, columns, upstream_script):
    """Guards against a run where an upstream step (upstream_script)
    didn't complete and therefore never wrote a column this script
    depends on (clean_text / actors_mentioned) - previously an unhandled
    KeyError partway through a batch; now a clean SKIP_REASON + exit(3)
    up front, which run_pipeline.py's run_step() recognizes as a graceful
    skip (same convention as preprocess_x_text.py's own NLTK self-heal
    skip). Added after a real scheduled-run incident where
    preprocess_x_text.py failed and this script (and others) then crashed
    with KeyError: 'clean_text' instead of skipping cleanly."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        print(
            f"SKIP_REASON: column(s) {missing} not found in {X_DATA_CSV} - "
            f"{upstream_script} has not completed successfully yet this cycle. "
            f"Nothing to do until it does; will retry next cycle."
        )
        sys.exit(3)


def load_data():
    df = pd.read_csv(X_DATA_CSV, dtype={"id": str})
    require_columns(df, ["clean_text", "actors_mentioned"],
                     "preprocess_x_text.py / extract_actor_mentions.py")
    # relations_recheck_done (v2.3): tracks which rows have already been queued for
    # re-extraction under the tightened prompt, so a second --recheck-relations
    # invocation doesn't re-purge/re-reset rows it already handed off - see the
    # v2.3 docstring section and --recheck-relations's handling in main().
    for col in ("topic_relevant", "relations_checked", "relations_recheck_done"):
        if col not in df.columns:
            df[col] = pd.array([pd.NA] * len(df), dtype="boolean")
        else:
            df[col] = df[col].astype("boolean")
    return df


def save_checkpoint(df):
    tmp_path = X_DATA_CSV + ".tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, X_DATA_CSV)


def append_relations(rows):
    """Appends new (post_id, actor_1, actor_2, relation_type) rows to
    RELATIONS_CSV - but first checks each one against every (post_id,
    actor_1, actor_2) combination already in the file, and skips any row
    that's already present, rather than blindly appending.

    Added 2026-09-08 as a hard safeguard after a real incident: a bug in
    clean_x_data.py (a separate script, now fixed - see its module
    docstring) silently reset relations_checked/topic_relevant to blank on
    every pipeline run, making this script believe already-processed posts
    were untouched. It then re-sent them to Groq and this function
    dutifully appended their relations again - one real post
    (2091511994381685126) was reprocessed 8 times, sometimes with
    contradictory classifications across passes (e.g. "neutral-reporting"
    then later "hostile" for the same Iran<->United States pair), leaving
    2,021 of 4,803 rows (42%) as duplicates that corrupted downstream
    network-centrality/edge-weight calculations.

    This check is deliberately a safeguard of last resort, not the primary
    fix - the clean_x_data.py bug is what actually caused the duplication,
    and that's fixed at the source. But a hard "never write the same
    (post_id, actor_1, actor_2) combination twice" guarantee here means
    even an UNKNOWN future bug that causes a post to be re-submitted can't
    silently corrupt actor_relations.csv again - the worst case becomes a
    skipped row and a printed note, not silent duplication.

    Deliberately does NOT try to detect a post whose relation_type differs
    from a prior pass for the same (post_id, actor_1, actor_2) - that's a
    legitimate re-classification, and --recheck-relations already handles
    it correctly by purging a post's old rows via
    purge_relations_for_post_ids() BEFORE its freshly re-extracted
    relations are appended (see process_group's purge_before_batch), so by
    the time this function runs during a recheck, the old row for that
    combination is already gone and the new one is not actually a
    duplicate. This function only ever needs to catch the accidental case:
    the exact same combination submitted again with the OLD rows still
    present, which is exactly what the clean_x_data.py bug caused."""
    if not rows:
        return

    existing = set()
    relations_csv_exists = os.path.exists(RELATIONS_CSV) and os.path.getsize(RELATIONS_CSV) > 0
    if relations_csv_exists:
        with open(RELATIONS_CSV, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # header
            for row in reader:
                if len(row) >= 3:
                    existing.add((row[0], row[1], row[2]))

    new_rows = []
    skipped = []
    seen_this_call = set()  # also guard duplicates WITHIN this same batch
    for row in rows:
        key = (str(row[0]), str(row[1]), str(row[2]))
        if key in existing or key in seen_this_call:
            skipped.append(row)
            continue
        seen_this_call.add(key)
        new_rows.append(row)

    if skipped:
        print(f"  [DEDUP SAFEGUARD] append_relations: skipped {len(skipped)} row(s) whose "
              f"(post_id, actor_1, actor_2) already exists in {RELATIONS_CSV} - see the "
              f"2026-09-08 incident note in this function's docstring. Skipped: {skipped}")

    if not new_rows:
        return
    file_is_new = not relations_csv_exists
    with open(RELATIONS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if file_is_new:
            writer.writerow(["post_id", "actor_1", "actor_2", "relation_type"])
        writer.writerows(new_rows)


def purge_relations_for_post_ids(post_ids):
    """Rewrites RELATIONS_CSV, dropping every row whose post_id is in
    post_ids (string-compared). Used by --recheck-relations so stale
    relation rows produced under the old, over-permissive extraction
    prompt don't linger alongside freshly re-extracted ones for the same
    post. Returns the number of rows removed. No-op (returns 0) if the
    file doesn't exist yet. Atomic write via .tmp + os.replace, same
    pattern as save_checkpoint."""
    if not os.path.exists(RELATIONS_CSV):
        return 0
    post_id_set = {str(p) for p in post_ids}
    kept = []
    removed = 0
    with open(RELATIONS_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if row and row[0] in post_id_set:
                removed += 1
                continue
            kept.append(row)
    tmp_path = RELATIONS_CSV + ".tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        writer.writerows(kept)
    os.replace(tmp_path, RELATIONS_CSV)
    return removed


def log_failure(post_id, reason, raw):
    with open(FAILURE_LOG, "a", encoding="utf-8") as f:
        f.write(f"{post_id}\t{reason}\t{raw!r}\n")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def process_group(client, df, idx_list, mode, args, pacer, counters, purge_before_batch=False,
                   recheck_marker_col=None, otpm_pacer=None):
    """mode: "full" (ask relevance + relations) or "relations_only" (relevance
    already known True; ask relations only). Mutates df and counters in place,
    checkpointing after every batch.

    otpm_pacer (v2.4, optional): a second TokenPacer, paced against
    OTPM_LIMIT and fed only OUTPUT tokens, passed straight through to
    call_groq_with_retry alongside an estimated-output-tokens figure
    computed per batch below. See the v2.4 docstring section.

    purge_before_batch (v2.3.1, used by --recheck-relations only): if True,
    a post's EXISTING actor_relations.csv rows are purged only once that
    SPECIFIC post has a confirmed (error-free) result parsed back for this
    batch - never upfront for a whole target set, and never for a post whose
    batch failed to parse at all. This bounds the "old data deleted, new data
    not yet written" exposure window to at most one batch's worth of posts,
    and only the ones that already have a fresh result waiting to be applied
    - not the ones still waiting on a future batch. See the v2.3.1 docstring
    section for the incident this replaced (purging an entire 1,316-row
    target set upfront, before any of it was confirmed).

    recheck_marker_col (v2.3.1): if set, this column is set True on a row at
    the EXACT same moment relations_checked is set True for it - i.e. only
    after that specific row's result is confirmed, never upfront. Used by
    --recheck-relations to record "this post's current relations were
    produced under the tightened prompt" for future auditability, WITHOUT
    that column ever being trusted again as a resume/gating signal (that's
    what caused the v2.3.1 bug - see run_recheck_relations)."""
    if not idx_list:
        return
    batches = build_batches(idx_list, df, args.batch_size, args.max_batch_input_tokens)
    per_post_ceiling = PER_POST_COMPLETION_CEILING_FULL if mode == "full" else PER_POST_COMPLETION_CEILING_RELATIONS

    for b_num, batch_idx in enumerate(batches, 1):
        batch_rows, expected = [], {}
        for idx in batch_idx:
            post_id = str(df.at[idx, "id"])
            text = str(df.at[idx, "clean_text"])
            actors = parse_actors(df.at[idx, "actors_mentioned"])
            batch_rows.append({"post_id": post_id, "text": text, "actors": actors})
            expected[post_id] = {"actors": actors, "idx": idx}

        # v2.5: MAX_TOKENS_PER_REQUEST_HARD_CAP is always included here - this is what
        # actually prevents a single request from ever asking for >900, regardless of
        # batch size or how per_post_ceiling is configured. See the v2.5 docstring
        # section and the MAX_TOKENS_PER_REQUEST_HARD_CAP constant's comment.
        max_tokens = min(ABSOLUTE_MAX_COMPLETION_TOKENS, per_post_ceiling * len(batch_rows),
                          MAX_TOKENS_PER_REQUEST_HARD_CAP)
        messages = build_full_batch_messages(batch_rows) if mode == "full" else build_relations_only_batch_messages(batch_rows)
        prompt_estimate = sum(estimate_tokens(r["text"]) for r in batch_rows) + 250  # +system prompt overhead
        estimated_output = int(max_tokens * 0.5)  # assume ~half the ceiling typically used - also the
                                                    # pre-call OTPM estimate (v2.4), same assumption, output-only
        estimated_total = prompt_estimate + estimated_output

        label = f"{mode} batch {b_num}/{len(batches)} ({len(batch_rows)} posts)"
        raw = call_groq_with_retry(
            client, messages, max_tokens, label, pacer, estimated_total,
            response_format={"type": "json_object"},
            otpm_pacer=otpm_pacer, estimated_output_tokens=estimated_output,
        )
        counters["n_calls"] += 1

        results, batch_err = parse_batch_response(raw, expected, mode)
        if batch_err:
            log_failure("BATCH", batch_err, raw)
            print(f"  [WARN] {label}: whole batch failed to parse ({batch_err}) - all {len(batch_rows)} row(s) will retry next run")
            # purge_before_batch: nothing purged for this batch's posts - a batch that
            # never parsed leaves whatever relations these posts already had untouched.
            continue

        if purge_before_batch:
            confirmed_post_ids = [pid for pid, res in results.items() if res and not res.get("error")]
            if confirmed_post_ids:
                purge_relations_for_post_ids(confirmed_post_ids)

        for post_id, info in expected.items():
            idx = info["idx"]
            res = results.get(post_id)
            if res is None or res.get("error"):
                reason = res.get("error") if res else "no result returned"
                log_failure(post_id, reason, raw)
                print(f"  [WARN] {label}: {post_id} - {reason}, will retry next run")
                continue

            if mode == "full":
                df.at[idx, "topic_relevant"] = res["topic_relevant"]

            relevant = bool(df.at[idx, "topic_relevant"]) if not pd.isna(df.at[idx, "topic_relevant"]) else False
            actors = info["actors"]
            relations = res["relations"] or []
            if relevant and len(actors) >= 2 and relations:
                append_relations([(post_id, a1, a2, rtype) for a1, a2, rtype in relations])
            df.at[idx, "relations_checked"] = True
            if recheck_marker_col:
                df.at[idx, recheck_marker_col] = True
            counters["n_rows_done"] += 1

        save_checkpoint(df)
        elapsed = time.monotonic() - counters["t_start"]
        print(f"[checkpoint] {label} done - {counters['n_rows_done']} row(s) total, "
              f"{counters['n_calls']} call(s), {elapsed:.0f}s elapsed")


def run_recheck_relations(client, df, en_mask, args):
    """v2.3.1 FIX (see the v2.3.1 docstring section for the full incident):
    the first version of this function marked relations_recheck_done=True
    for its ENTIRE target set (all 1,316 topic_relevant=True rows) and
    purged ALL of their actor_relations.csv rows upfront, before a single
    Groq call was made - so a run interrupted by a rate wall left most
    targeted posts with their old relations deleted and nothing to replace
    them, while relations_recheck_done still (wrongly) said "already
    handled" for every one of them. A follow-up --recheck-relations then
    saw nothing left to target and reported "nothing left to recheck" -
    which was technically true of that (broken) flag, but not of the real
    data. Confirmed against the real files: only 344 of 1,316 targeted
    rows actually got reprocessed; the other 972 were left with zero
    relation rows.

    This version trusts ONLY relations_checked - already proven accurate,
    since it is set exactly once per row, exactly when that row's Groq
    result is confirmed - to decide what still needs work. It does NOT
    read relations_recheck_done for target selection at all; that column
    is still written (see process_group's recheck_marker_col), but purely
    for future auditability, and only at the same per-row confirmed moment
    as relations_checked, via purge_before_batch=True/recheck_marker_col
    passed to process_group - never upfront. This also means the CURRENT
    (wrong) relations_recheck_done values already on disk (True for all
    1,316 targeted rows, only 344 of which are real) are simply never
    consulted again - nothing needs to be manually reset. As the 972
    outstanding rows get correctly reprocessed by this fixed version, this
    column will self-correct to an accurate state alongside them."""
    target_mask = en_mask & (df["topic_relevant"].fillna(False) == True) & df["relations_checked"].isna()  # noqa: E712
    target_idx = list(df.index[target_mask])

    if args.limit:
        target_idx = target_idx[: args.limit]

    if not target_idx:
        print(
            "Nothing left to recheck - every English row with topic_relevant=True already has "
            "relations_checked=True (the only signal this trusts for what's outstanding)."
        )
        return

    print(
        f"--recheck-relations: {len(target_idx)} row(s) targeted (topic_relevant=True, "
        f"relations_checked not yet True). A post's stale actor_relations.csv rows are purged "
        f"only once THAT post's fresh result is confirmed - never upfront for the whole target set - "
        f"and relations_checked/relations_recheck_done are set True only at that same confirmed "
        f"moment, per post. topic_relevant is NOT being re-asked."
    )

    pacer = TokenPacer(TPM_LIMIT)
    otpm_pacer = TokenPacer(OTPM_LIMIT, safety_factor=OTPM_SAFETY_FACTOR)  # v2.4 - see docstring section
    counters = {"n_calls": 0, "n_rows_done": 0, "t_start": time.monotonic()}
    process_group(client, df, target_idx, "relations_only", args, pacer, counters,
                   purge_before_batch=True, recheck_marker_col="relations_recheck_done",
                   otpm_pacer=otpm_pacer)

    elapsed = time.monotonic() - counters["t_start"]
    print(f"\nRecheck done. {counters['n_calls']} batched Groq call(s) covering "
          f"{counters['n_rows_done']} row(s), {elapsed:.0f}s elapsed.")
    if os.path.exists(FAILURE_LOG):
        print(f"Some rows failed to parse or were dropped by the model - see {FAILURE_LOG}. "
              f"Just re-run --recheck-relations (or a plain run with no flags) to retry those specific "
              f"rows - relations_checked is still NA for them, so they'll be picked up automatically "
              f"either way.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                         help="only process the first N rows that still need work (smoke test)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT,
                         help=f"max posts per API call (default {BATCH_SIZE_DEFAULT})")
    parser.add_argument("--max-batch-input-tokens", type=int, default=MAX_BATCH_INPUT_TOKENS_DEFAULT,
                         help=f"also cut a batch short if estimated input tokens would exceed this "
                              f"(default {MAX_BATCH_INPUT_TOKENS_DEFAULT})")
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE_DEFAULT,
                         help=f"target TOTAL done-row count (already-done rows + newly-processed this "
                              f"run), prioritizing highest (likes+retweets) engagement among rows still "
                              f"outstanding. Already-done rows are never discarded even if they fall "
                              f"outside the sample. 0 disables sampling and processes every remaining "
                              f"row (default {SAMPLE_SIZE_DEFAULT}). IGNORED when --recheck-relations "
                              f"is set (recheck always targets every not-yet-rechecked topic_relevant=True "
                              f"row, or the first --limit of them for a smoke test).")
    parser.add_argument("--recheck-relations", action="store_true",
                         help="v2.3.1: re-run RELATION EXTRACTION ONLY (never relevance/topic_relevant) "
                              "for every row with topic_relevant=True whose relations_checked is not "
                              "yet True - that single column is the ONLY signal this trusts for what's "
                              "outstanding (see the v2.3.1 fix in this file's module docstring - an "
                              "earlier version used a separate tracking column that got marked True "
                              "upfront, before work was confirmed, and produced false 'nothing left to "
                              "recheck' reports after an interrupted run). --sample-size is IGNORED "
                              "under this flag - every outstanding row is always in scope, regardless "
                              "of sample size. A post's stale actor_relations.csv rows are purged only "
                              "once THAT post's fresh result is confirmed, never upfront for the whole "
                              "target set, so an interrupted run never deletes more than it replaces. "
                              "Safe to interrupt and resume: just run --recheck-relations again (or a "
                              "plain run with no flags) - relations_checked accurately reflects exactly "
                              "what's left either way. Combine with --limit for a smoke test first.")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        sys.exit("GROQ_API_KEY not found in .env - add it and try again.")
    client = Groq(api_key=api_key, max_retries=0)  # we handle retries ourselves, for visible logging

    df = load_data()
    en_mask = df["detected_language"] == "en"

    if args.recheck_relations:
        run_recheck_relations(client, df, en_mask, args)
        return

    in_scope_idx, already_done_count, newly_sampled_count = build_sample_idx(df, en_mask, args.sample_size)

    needs_relevance = df["topic_relevant"].isna()
    needs_relations = (df["topic_relevant"].fillna(False) == True) & df["relations_checked"].isna()  # noqa: E712
    todo_full_idx = [i for i in df.index[en_mask & needs_relevance] if i in in_scope_idx]
    todo_relations_only_idx = [i for i in df.index[en_mask & needs_relations] if i in in_scope_idx]

    if args.limit:
        if len(todo_full_idx) >= args.limit:
            todo_full_idx = todo_full_idx[: args.limit]
            todo_relations_only_idx = []
        else:
            todo_relations_only_idx = todo_relations_only_idx[: args.limit - len(todo_full_idx)]

    total_en = int(en_mask.sum())
    sample_note = ""
    if args.sample_size and args.sample_size > 0:
        left_out = total_en - already_done_count - newly_sampled_count
        sample_note = (
            f" [sampling active: target {args.sample_size} total done rows - {already_done_count} "
            f"already done (kept as-is) + {newly_sampled_count} newly selected by (likes+retweets) "
            f"engagement this run; {left_out} lower-engagement row(s) left untouched for now]"
        )
    print(f"{len(todo_full_idx)} row(s) need relevance+relations, {len(todo_relations_only_idx)} row(s) "
          f"need relations only (of {total_en} English rows total).{sample_note}")
    if not todo_full_idx and not todo_relations_only_idx:
        print("Nothing to do at the current --sample-size" +
              (" - already-done rows meet or exceed the target. Raise --sample-size to process more."
               if args.sample_size and args.sample_size > 0 and already_done_count >= args.sample_size
               else " - every English row in scope already has topic_relevant and relations_checked set."))
        return

    pacer = TokenPacer(TPM_LIMIT)
    otpm_pacer = TokenPacer(OTPM_LIMIT, safety_factor=OTPM_SAFETY_FACTOR)  # v2.4 - see docstring section
    counters = {"n_calls": 0, "n_rows_done": 0, "t_start": time.monotonic()}

    process_group(client, df, todo_full_idx, "full", args, pacer, counters, otpm_pacer=otpm_pacer)
    process_group(client, df, todo_relations_only_idx, "relations_only", args, pacer, counters, otpm_pacer=otpm_pacer)

    elapsed = time.monotonic() - counters["t_start"]
    print(f"\nDone. {counters['n_calls']} batched Groq call(s) covering {counters['n_rows_done']} row(s), "
          f"{elapsed:.0f}s elapsed.")
    if os.path.exists(FAILURE_LOG):
        print(f"Some rows failed to parse or were dropped by the model - see {FAILURE_LOG}. "
              f"Just re-run this script to retry those rows.")


if __name__ == "__main__":
    main()

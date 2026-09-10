"""
run_pipeline.py

Scheduled background collection pipeline for the Makkah/Mecca pact project,
run INDEPENDENTLY of the Streamlit dashboard (app.py). Per the collection/
serving split: the dashboard only ever READS existing files now - it never
fetches, cleans, or calls any API itself. This script is the one place new
data enters the project on an ongoing basis (outside of running a single
step by hand for debugging).

--- Chain (in order) ---
  1. fetch_x_data.py --recent           pull new posts from X (small, fast,
                                         rate-limited "what's new" pull -
                                         NOT the full historical backfill)
  2. clean_x_data.py                    incremental cleaning (dedup
                                         clustering / language detection /
                                         flags only run on genuinely new rows)
  3. preprocess_x_text.py               clean_text / lemmatization, English subset
  4. extract_topics_hashtags.py         hashtags + actor-match verification, English subset
  5. extract_actor_mentions.py          actors_mentioned field, English subset
  6. add_sentiment.py                   VADER sentiment, English subset
  7. relevance_and_relations.py --sample-size 0
                                         topic_relevant + relation extraction,
                                         but ONLY for rows that don't already
                                         have it (i.e. genuinely new rows from
                                         steps 1-6 above) - never a full
                                         recheck. --sample-size 0 is REQUIRED,
                                         not optional: the script's own
                                         default sample-size (400) is already
                                         smaller than the ~1,316 rows already
                                         done, so leaving it at default would
                                         make the script report "nothing to
                                         do" and silently skip every new row
                                         every single cycle. See that file's
                                         own --sample-size help text.
  8. build_network_v2.py                typed, weighted actor_network_v2.json
                                         + actor_centrality.csv (networkx
                                         centrality), rebuilt from
                                         actor_relations.csv.
  9. build_daily_timelines.py --x-only  daily_actor_timeline.csv (the
                                         Timeline tab's data). NOT in the
                                         originally-dictated chain for this
                                         script, but the Timeline tab reads
                                         this file and nothing else produces
                                         it - added back in here so that tab
                                         doesn't go stale. --x-only skips
                                         GDELT, which this pipeline never
                                         re-fetches (unchanged from the old
                                         refresh_worker.py's own behavior).
  10. sync_to_git()                     commits + pushes ONLY the data files
                                         app.py actually reads (see SYNC_FILES
                                         below) to the GitHub repo backing the
                                         Streamlit Community Cloud deployment.
                                         See "Cloud deployment sync" below for
                                         why this exists and what it does NOT
                                         do (never touches code files).

--- Cloud deployment sync ---
This project is deployed on Streamlit Community Cloud, which runs in ITS
OWN cloud filesystem, pulling from a connected GitHub repo - completely
separate from this machine. Without this step, everything above would keep
updating files HERE, on this machine, with zero effect on what a deployed
viewer sees: the cloud app only ever reflects whatever was last pushed to
its tracked branch.

sync_to_git() bridges that gap the standard way for Streamlit Community
Cloud: `git add` the exact files app.py reads (SYNC_FILES below) -> `git
commit` (a no-op, logged as a clean success, if nothing actually changed
this cycle - e.g. every step above was skipped) -> `git push`. Streamlit
Community Cloud watches its tracked branch and auto-redeploys on push,
typically within a minute or two - not instant, but close enough for an
"every few hours" schedule.

What this step deliberately does NOT do:
  - It never touches CODE files (app.py, run_pipeline.py, build_network_v2.py,
    relevance_and_relations.py, requirements.txt, etc.). Pushing code
    changes is a deliberate, reviewed, manual action - never something an
    unattended scheduled task should do silently. If you edit any of those,
    commit and push them yourself as normal.
  - It never touches x_data.csv (the raw, un-cleaned pull), accounts.db, or
    .env - already excluded from the repo by .gitignore, and this step
    doesn't override that.
  - It does not fetch, pull, or rebase first. This assumes the local clone
    stays ahead of (or in sync with) origin/main - true as long as nothing
    else pushes to this repo from elsewhere. If a push is ever rejected for
    being behind, this logs FAILED with git's own message rather than
    guessing at a merge.

Prerequisite (one-time, manual, cannot be done by this script or by
Claude): a GitHub Personal Access Token (PAT) with write access to this
repo, stored as GITHUB_PAT=... in .env (see the setup notes delivered
alongside this file for exactly how to generate one and what scope to
give it). This step authenticates each push itself, in-process, using
GITHUB_PAT - it never relies on Git Credential Manager or any cached
interactive login, which is what makes it safe to run unattended under
Task Scheduler. If GITHUB_PAT isn't set yet, this step logs a clean SKIP
(not a FAILED) every run - harmless, nothing else in the pipeline is
affected - but the deployed dashboard's data stays frozen until it's
added. The token is read from .env at call time only, is never written to
.git/config (the authenticated URL is passed directly to `git push`, not
saved via `git remote set-url`), and any git output that might echo it
back (some git versions include the remote URL, credentials and all, in
their own error text) is redacted before it's ever logged or printed.

Also worth knowing: x_data_cleaned.csv (raw post text, scraped from X) is
part of SYNC_FILES because the Sentiment/Data Explorer tabs need it to
work on the deployed version - but that means this pushes real scraped
post text into this GitHub repo's history on an ongoing basis. Fine for a
private repo; worth a second thought if this repo is (or ever becomes)
public. Remove "x_data_cleaned.csv" from SYNC_FILES below if you'd rather
the deployed dashboard not carry raw post text, at the cost of the
Sentiment/Explorer tabs going stale on the deployed version specifically
(they'd keep working fine when run locally, reading the always-current
local file).

Deliberately DROPPED from the old refresh_worker.py chain this replaces:
  - dedupe_actor_cooccurrence.py and the OLD build_actor_network.py
    (co-occurrence-based network). Both only ever fed the OLD
    actor_network.json, which the dashboard no longer reads (the Actor
    Network tab now reads actor_network_v2.json / actor_centrality.csv,
    built directly from actor_relations.csv - see build_network_v2.py).
    Running them would just spend time computing files nothing uses
    anymore. Say the word if you still want them kept for some other
    reason and I'll add them back.

--- Graceful degradation (the whole point of this rewrite) ---
Every step above runs in its own try/except via run_step() below, which
NEVER raises - it always returns a result dict, so main() just keeps going
no matter what happened. Concretely:
  - X collection failing or getting rate-limited doesn't stop anything -
    steps 2-9 run against whatever's already in x_data.csv regardless ("0
    new rows" is itself a normal, common, non-error outcome, not a failure).
  - relevance_and_relations.py hitting a Groq TPD/RPD/TPM/OTPM/rate-limit
    wall is specifically detected (by scanning its crash output for the
    same limit-type keywords the script itself already reports - see
    QUOTA_KEYWORDS below) and logged as a clean SKIP with the message
    "Relation extraction skipped this cycle - Groq quota limit, will retry
    next cycle" - not a generic failure. build_network_v2.py still runs
    right after, using whatever actor_relations.csv already has (it just
    won't reflect this cycle's newest rows yet, until a future cycle's
    relation extraction succeeds).
  - Any OTHER step failing (bad exit code, exception, timeout) is logged as
    FAILED with a short excerpt of what went wrong, and the chain still
    continues to the next step. No single step's failure ever leaves the
    pipeline half-finished or crashes the whole run.

--- Status log ---
Writes pipeline_status.log (appended, never truncated/rotated) - one block
per run: a line per step (SUCCESS/SKIPPED/FAILED + a short detail) followed
by one machine-parseable summary line, PIPE-separated (not colon-separated
- an ISO timestamp itself contains colons, e.g. "2026-09-06T18:01:14Z",
which silently broke an earlier colon-delimited version of this line - a
real bug caught by actually rendering the dashboard against a sample log
rather than just eyeballing the format):
    PIPELINE_RUN|<iso timestamp>|<success|partial|failed>|<note>
app.py's dashboard header reads ONLY that last summary line for its "Data
last updated" display - not any single file's mtime, which could be
misleading if a step partially failed (e.g. clean_x_data.py's own output
file's mtime would look "fresh" even on a run where relation extraction
was skipped for a quota wall).

Run manually:
    python run_pipeline.py
Run on a schedule: see run_pipeline.bat plus the Windows Task Scheduler
setup notes delivered alongside this file.
"""

import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
STATUS_LOG = os.path.join(DATA_DIR, "pipeline_status.log")

FETCH_X_SCRIPT = os.path.join(DATA_DIR, "fetch_x_data.py")
CLEAN_SCRIPT = os.path.join(DATA_DIR, "clean_x_data.py")
PREPROCESS_SCRIPT = os.path.join(DATA_DIR, "preprocess_x_text.py")
HASHTAGS_SCRIPT = os.path.join(DATA_DIR, "extract_topics_hashtags.py")
ACTOR_MENTIONS_SCRIPT = os.path.join(DATA_DIR, "extract_actor_mentions.py")
SENTIMENT_SCRIPT = os.path.join(DATA_DIR, "add_sentiment.py")
RELATIONS_SCRIPT = os.path.join(DATA_DIR, "relevance_and_relations.py")
NETWORK_V2_SCRIPT = os.path.join(DATA_DIR, "build_network_v2.py")
TIMELINE_SCRIPT = os.path.join(DATA_DIR, "build_daily_timelines.py")

# Exactly the files app.py itself opens (see that file's own DATA_DIR
# constants) - nothing more. Deliberately excludes every code file and
# every raw/credentialed file (x_data.csv, accounts.db, .env) - see
# "Cloud deployment sync" in the module docstring above for the full
# reasoning, including the x_data_cleaned.csv privacy consideration.
SYNC_FILES = [
    "x_data_cleaned.csv",
    "actor_network_v2.json",
    "actor_centrality.csv",
    "daily_actor_timeline.csv",
    "daily_gdelt_timeline.csv",
    "hashtag_frequency.csv",
    "pipeline_status.log",
]
GIT_TIMEOUT_SEC = 3 * 60

# True now that the dashboard is deployed on Streamlit Community Cloud and
# this step has a real target to push to. Once GITHUB_PAT is added to .env
# (see the module docstring's "Prerequisite" section and the setup notes
# delivered alongside this file), sync_to_git() authenticates and pushes
# on its own - no cached credential, no interactive login.
#
# Until GITHUB_PAT is actually added, sync_to_git() logs a clean SKIPPED
# every run (not a FAILED) - which does mean pipeline_status.log's overall
# status shows "partial" instead of "success" during that interim window,
# purely because of this one step. Expected, not a bug: add GITHUB_PAT and
# it clears up on the next run. (Set this back to False if you ever want
# to stop pushing to GitHub entirely without touching anything else below.)
ENABLE_GIT_SYNC = True

NEW_ROWS_RE = re.compile(r"NEW_ROWS_ADDED:(-?\d+)")

# General-purpose "I have nothing to do this cycle, that's fine, don't
# treat me as broken" convention: any step script can print a line
# starting with SKIP_REASON: and exit with code 3 to signal a clean skip.
# run_step() below checks for this BEFORE the quota-wall check, for every
# step, not just quota_aware ones. Introduced after a real scheduled-run
# failure where preprocess_x_text.py crashed with an unhandled NLTK
# LookupError, and every downstream step then crashed too with
# KeyError: 'clean_text' / 'actors_mentioned' - loud, confusing FAILED
# entries for what was really one root cause. preprocess_x_text.py and
# every script downstream of it that depends on clean_text/
# actors_mentioned now use this convention: they check their required
# input up front and exit(3) with a clear one-line reason instead of
# letting a KeyError (or anything else) propagate.
SKIP_REASON_RE = re.compile(r"^SKIP_REASON:\s*(.+)$", re.MULTILINE)
SKIP_EXIT_CODE = 3

# Keywords that show up in relevance_and_relations.py's own RuntimeError
# messages when it hits a Groq DAILY or PER-MINUTE quota wall (TPD/RPD/TPM/
# OTPM) or a 429 rate-limit response generally - see that file's own
# call_groq_with_retry() for exactly where these get raised. If a crash's
# combined stdout+stderr contains any of these, treat it as a quota wall,
# not a bug - log it as a clean SKIP instead of a FAILED.
QUOTA_KEYWORDS = ["TPD", "RPD", "TPM", "OTPM", "RateLimitError", "429", "rate limit", "quota"]

# Every FAILED step's FULL captured stdout+stderr is appended here (never
# truncated/rotated, same as pipeline_status.log). Added after a real
# incident where the only trace of a failure was a 200-character-truncated
# last line of output that happened to be a row of '*' characters (NLTK's
# LookupError formats its message inside a banner bordered by asterisk
# rows - see preprocess_x_text.py's module docstring) - the actual
# exception and traceback had already been discarded by the time anyone
# went looking for it. pipeline_status.log stays a short, glanceable
# summary; this file is where the real diagnostic text lives.
STEP_ERROR_LOG = os.path.join(DATA_DIR, "pipeline_step_errors.log")

# Per-step timeouts. Unlike the old interactive refresh_worker.py (no
# timeout at all, because a human was watching a live log and could just
# wait it out), this now runs unattended on a schedule - a stuck step must
# never be allowed to block forever and delay or collide with the next
# scheduled run. These are generous enough for real rate-limit pacing
# delays while staying well under a "every few hours" schedule interval.
DEFAULT_TIMEOUT_SEC = 20 * 60
FETCH_TIMEOUT_SEC = 15 * 60       # --recent mode only, a small pull
RELATIONS_TIMEOUT_SEC = 45 * 60   # OTPM pacing (1000 output tokens/min) can genuinely take a while


def _now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_banner_line(ln):
    """A line made of nothing but one repeated border character (NLTK's
    LookupError borders its message with rows of 70 '*' characters)."""
    s = ln.strip()
    return len(s) >= 3 and len(set(s)) == 1 and s[0] in "*=-#"


def _extract_error_reason(output):
    """Picks the most useful line out of a failed command's combined
    stdout+stderr, for a short one-line summary. The literal LAST line is
    often NOT the useful one:
      - A rejected git push prints its real reason ("! [rejected] ..." /
        "error: failed to push...") followed by several trailing "hint:"
        lines pointing at `git push --help`.
      - An unhandled NLTK LookupError formats its message inside a banner
        bordered by rows of '*' characters, with Python's own "LookupError:"
        traceback line printed EMPTY (the real text starts on the line
        after it) - so neither "the exception line" nor "the last line"
        is the actual message; it's the first real content line after the
        (empty) exception header.
    All caught by testing the actual failure, not by inspection. Tries, in
    order:
      1. A "SomeError:"/"SomeException:" line WITH real text after the
         colon on that same line - use it directly.
      2. A "SomeError:"/"SomeException:" line with NOTHING after the colon
         (the NLTK shape) - use the first non-blank, non-banner line that
         follows it instead.
      3. The last "error:" line (git's shape).
      4. The last line mentioning "[rejected]" (git's shape).
      5. The last line that isn't a trailing "hint:" line or a pure
         banner row.
      6. The literal last line, if nothing above matched.
    """
    raw_lines = output.strip().splitlines()
    lines = [ln for ln in raw_lines if ln.strip()]
    if not lines:
        return None

    exception_re = re.compile(r"^(\w*(?:Error|Exception))\s*:\s*(.*)$")
    last_exception_idx = None
    for i, ln in enumerate(raw_lines):
        if exception_re.match(ln.strip()):
            last_exception_idx = i
    if last_exception_idx is not None:
        m = exception_re.match(raw_lines[last_exception_idx].strip())
        if m.group(2).strip():
            return raw_lines[last_exception_idx].strip()[:200]
        for ln in raw_lines[last_exception_idx + 1:]:
            if ln.strip() and not _is_banner_line(ln):
                return ln.strip()[:200]

    for is_match in (
        lambda ln: ln.strip().startswith("error:"),
        lambda ln: "[rejected]" in ln,
        lambda ln: not ln.strip().startswith("hint:") and not _is_banner_line(ln),
    ):
        matches = [ln for ln in lines if is_match(ln)]
        if matches:
            return matches[-1].strip()[:200]
    return lines[-1].strip()[:200]


def write_step_error_log(label, iso_ts, returncode, output):
    """Appends the FULL captured stdout+stderr of a FAILED step to
    STEP_ERROR_LOG - never truncated, unlike the short `detail` string
    that goes into pipeline_status.log. See STEP_ERROR_LOG's own comment
    above for why this exists (a real incident where the true error was
    silently discarded and unrecoverable after the fact)."""
    block = (
        f"===== {label} FAILED at {iso_ts} (exit code {returncode}) =====\n"
        f"{output.rstrip()}\n"
        f"===== end {label} =====\n\n"
    )
    with open(STEP_ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(block)


def run_step(script_path, label, extra_args=None, timeout_sec=DEFAULT_TIMEOUT_SEC, quota_aware=False):
    """Run one pipeline step as its own subprocess. NEVER raises - always
    returns a result dict {"label", "status", "detail", "elapsed"} with
    status in {"SUCCESS", "SKIPPED", "FAILED"}, so main() can just keep
    going regardless of what happened. This is the one function
    responsible for "no single step's failure crashes the whole chain."
    """
    if not os.path.exists(script_path):
        return {"label": label, "status": "FAILED",
                "detail": f"{os.path.basename(script_path)} not found", "elapsed": 0.0}

    args = [sys.executable, script_path] + (extra_args or [])
    start = time.monotonic()
    try:
        # encoding="utf-8", errors="replace" - explicit, rather than relying
        # on text=True's default (locale.getpreferredencoding(), cp1252 on
        # a typical Windows machine). Added after a real incident where a
        # step script printed scraped X/Twitter text containing an emoji:
        # any step here processes that kind of raw text, so any of them
        # could in principle write a byte sequence the parent's default
        # locale can't decode. errors="replace" means a stray undecodable
        # byte becomes a placeholder character in the captured output
        # instead of this subprocess.run() call itself raising - this
        # orchestrator must never crash because of what a child step
        # happened to print.
        proc = subprocess.run(
            args, cwd=DATA_DIR, capture_output=True, text=True, timeout=timeout_sec,
            encoding="utf-8", errors="replace",
        )
        elapsed = time.monotonic() - start
        output = (proc.stdout or "") + (proc.stderr or "")

        if proc.returncode == 0:
            new_rows_match = NEW_ROWS_RE.search(output)
            detail = f"{new_rows_match.group(1)} new row(s)" if new_rows_match else ""
            return {"label": label, "status": "SUCCESS", "detail": detail, "elapsed": elapsed}

        # Non-zero exit. Check, in order: (1) the general-purpose clean-skip
        # convention (any step can use this, not just quota_aware ones -
        # see SKIP_REASON_RE's comment above), (2) quota_aware steps
        # specifically for a Groq quota wall, (3) otherwise a genuine
        # failure - full output preserved to STEP_ERROR_LOG, short reason
        # extracted for the compact status log.
        if proc.returncode == SKIP_EXIT_CODE:
            match = SKIP_REASON_RE.search(output)
            detail = match.group(1).strip()[:200] if match else (
                "step exited with the clean-skip code (3) but printed no SKIP_REASON line"
            )
            return {"label": label, "status": "SKIPPED", "detail": detail, "elapsed": elapsed}

        if quota_aware and any(kw in output for kw in QUOTA_KEYWORDS):
            matched = next(kw for kw in QUOTA_KEYWORDS if kw in output)
            return {
                "label": label, "status": "SKIPPED",
                "detail": f"Relation extraction skipped this cycle - Groq quota limit "
                          f"({matched}), will retry next cycle",
                "elapsed": elapsed,
            }

        reason = _extract_error_reason(output) or f"exit code {proc.returncode}"
        write_step_error_log(label, _now_iso(), proc.returncode, output)
        return {"label": label, "status": "FAILED", "detail": reason, "elapsed": elapsed}

    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - start
        partial_output = (exc.stdout or "") + (exc.stderr or "") if (exc.stdout or exc.stderr) else ""
        if partial_output.strip():
            write_step_error_log(label, _now_iso(), "TIMEOUT", partial_output)
        return {"label": label, "status": "FAILED",
                "detail": f"timed out after {timeout_sec}s", "elapsed": elapsed}
    except Exception as exc:
        elapsed = time.monotonic() - start
        return {"label": label, "status": "FAILED", "detail": str(exc)[:200], "elapsed": elapsed}


def write_status_log(results, run_start_iso, total_elapsed):
    lines = [f"===== RUN START {run_start_iso} ====="]
    for r in results:
        detail = f" - {r['detail']}" if r["detail"] else ""
        lines.append(f"{r['label']:26s}: {r['status']:8s} ({r['elapsed']:.0f}s){detail}")

    n_ok = sum(1 for r in results if r["status"] == "SUCCESS")
    n_skipped = sum(1 for r in results if r["status"] == "SKIPPED")
    n_failed = sum(1 for r in results if r["status"] == "FAILED")
    # "success" only when literally every step completed clean; "partial"
    # when at least one step ran but something was skipped/failed; "failed"
    # only in the (should be rare, given every step is individually caught)
    # case that NOTHING succeeded at all.
    overall = "success" if (n_failed == 0 and n_skipped == 0) else ("partial" if n_ok > 0 else "failed")

    notable = [f"{r['label']}: {r['detail']}" for r in results if r["status"] in ("SKIPPED", "FAILED") and r["detail"]]
    note = "; ".join(notable) if notable else "all steps completed"

    finish_iso = _now_iso()
    lines.append(
        f"===== RUN END {finish_iso} | overall={overall} ok={n_ok} skipped={n_skipped} "
        f"failed={n_failed} elapsed={total_elapsed:.0f}s ====="
    )
    # This EXACT format is the one line app.py parses (see PIPELINE_RUN_RE
    # there) for its "Data last updated" header - PIPE-separated on
    # purpose, not colon-separated: finish_iso itself contains colons
    # (e.g. "2026-09-06T18:01:14Z"), which broke an earlier version of
    # this line's parsing. Keep this shape stable if edited again.
    lines.append(f"PIPELINE_RUN|{finish_iso}|{overall}|{note}")
    lines.append("")

    with open(STATUS_LOG, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# Matches a GitHub PAT embedded in an https:// URL, e.g.
# "https://ghp_xxx@github.com/owner/repo.git" - used to scrub the token out
# of any git output before it's ever printed or written to a log file. Some
# git versions echo the exact remote URL (credentials included) back in
# their own error/hint text on a failed push, so this is applied to every
# piece of git output this function might log, not just the push step.
_CREDENTIAL_URL_RE = re.compile(r"https://[^@\s]+@github\.com")


def _redact(text):
    return _CREDENTIAL_URL_RE.sub("https://***@github.com", text or "")


def sync_to_git(results):
    """Commits + pushes SYNC_FILES to the GitHub repo backing the deployed
    Streamlit Community Cloud app. See "Cloud deployment sync" in the module
    docstring above for the full reasoning. NEVER raises - always returns a
    result dict in the same {"label", "status", "detail", "elapsed"} shape
    as run_step()'s results, so main() can log it exactly like any other
    step and nothing downstream needs to treat it specially.

    Runs, in order: check GITHUB_PAT is set (SKIP cleanly if not) -> `git
    add` (only the SYNC_FILES that actually exist right now) -> `git
    commit` -> `git push` to an authenticated URL built from GITHUB_PAT,
    never from a cached credential. "nothing to commit" is a normal,
    common SUCCESS (a cycle where nothing changed - e.g. 0 new rows and
    every downstream step was a no-op), not a FAILED - this is also how
    "only push if something actually changed" is enforced: no commit means
    push is never even attempted. Any other git error is logged as FAILED
    with git's own (redacted) message, but since this always runs LAST, a
    failure here never affects anything else already written to disk this
    cycle; it only means the deployed app stays on stale data until it's
    fixed.

    The GitHub PAT itself is never written to disk by this function (not
    to .git/config, not to any log) and never appears in a SUCCESS detail
    string. It exists in memory only for the duration of this call, and
    only ever touches the single `git push` subprocess argument list that
    needs it.
    """
    label = "sync_to_github"
    start = time.monotonic()

    def _git(args):
        return subprocess.run(
            ["git"] + args, cwd=DATA_DIR, capture_output=True, text=True, timeout=GIT_TIMEOUT_SEC,
        )

    try:
        # load_dotenv() here (not at module scope) matches this project's
        # existing convention (see x_auth.py) - loaded fresh at call time
        # rather than assuming it happened once at import time.
        load_dotenv()
        github_pat = os.environ.get("GITHUB_PAT", "").strip()
        if not github_pat:
            elapsed = time.monotonic() - start
            return {"label": label, "status": "SKIPPED",
                    "detail": "GITHUB_PAT not set in .env - see the PAT setup notes to enable pushing",
                    "elapsed": elapsed}

        # Only add files that actually exist right now - e.g. the very
        # first-ever run, before build_network_v2.py has produced anything,
        # shouldn't make `git add` itself fail on a missing path.
        present_files = [f for f in SYNC_FILES if os.path.exists(os.path.join(DATA_DIR, f))]
        if not present_files:
            elapsed = time.monotonic() - start
            return {"label": label, "status": "SUCCESS",
                    "detail": "no sync files present yet, nothing to add", "elapsed": elapsed}

        add_proc = _git(["add"] + present_files)
        if add_proc.returncode != 0:
            elapsed = time.monotonic() - start
            detail = _extract_error_reason(_redact((add_proc.stderr or "") + (add_proc.stdout or ""))) or f"git add exit code {add_proc.returncode}"
            return {"label": label, "status": "FAILED", "detail": f"git add failed - {detail}", "elapsed": elapsed}

        # Short, informative commit message built from this run's own
        # results (e.g. how many new rows were fetched) rather than a
        # generic "update data" - so the commit history doubles as a
        # lightweight audit trail of what changed each cycle.
        new_rows_detail = next((r["detail"] for r in results if r["label"] == "fetch_x_data" and r["detail"]), None)
        commit_msg = "Automated data refresh"
        if new_rows_detail:
            commit_msg += f" ({new_rows_detail})"
        commit_msg += f" - {_now_iso()}"

        commit_proc = _git(["commit", "-m", commit_msg])
        elapsed = time.monotonic() - start
        if commit_proc.returncode != 0:
            combined = (commit_proc.stdout or "") + (commit_proc.stderr or "")
            # git's normal way of saying there were no changes to commit -
            # a real, common, non-error outcome, not a failure. This is
            # what makes the "only push if something changed" requirement
            # hold: nothing to commit means push is never reached below.
            if "nothing to commit" in combined or "nothing added to commit" in combined:
                return {"label": label, "status": "SUCCESS",
                        "detail": "no changes to sync this cycle", "elapsed": elapsed}
            detail = _extract_error_reason(_redact(combined)) or f"git commit exit code {commit_proc.returncode}"
            return {"label": label, "status": "FAILED", "detail": f"git commit failed - {detail}", "elapsed": elapsed}

        # Build a one-time authenticated push URL from the existing origin
        # remote, WITHOUT ever calling `git remote set-url` - the token is
        # passed straight to this one `git push` invocation and nowhere
        # else, so it's never written into .git/config and never shows up
        # in a later `git remote -v`.
        remote_proc = _git(["remote", "get-url", "origin"])
        if remote_proc.returncode != 0:
            elapsed = time.monotonic() - start
            detail = _extract_error_reason(_redact((remote_proc.stderr or "") + (remote_proc.stdout or ""))) or "could not read origin remote URL"
            return {"label": label, "status": "FAILED", "detail": f"git remote get-url failed - {detail}", "elapsed": elapsed}

        origin_url = remote_proc.stdout.strip()
        m = re.match(
            r"^(?:https://(?:[^@]+@)?github\.com/|git@github\.com:)([^/]+)/(.+?)(?:\.git)?/?$",
            origin_url,
        )
        if not m:
            elapsed = time.monotonic() - start
            return {"label": label, "status": "FAILED",
                    "detail": "origin remote is not a recognizable github.com URL - can't build an authenticated push URL",
                    "elapsed": elapsed}
        owner, repo = m.group(1), m.group(2)
        auth_url = f"https://{github_pat}@github.com/{owner}/{repo}.git"

        push_proc = _git(["push", auth_url, "HEAD:main"])
        elapsed = time.monotonic() - start
        if push_proc.returncode != 0:
            combined = _redact((push_proc.stdout or "") + (push_proc.stderr or ""))
            detail = _extract_error_reason(combined) or f"git push exit code {push_proc.returncode}"
            # Most likely causes: an expired/revoked PAT, the PAT missing
            # the right scope, a network hiccup, or (rare, since nothing
            # else is expected to push to this repo) origin/main having
            # moved since this clone's last known state - all logged the
            # same way here since git's own message already says which.
            return {"label": label, "status": "FAILED", "detail": f"git push failed - {detail}", "elapsed": elapsed}

        return {"label": label, "status": "SUCCESS", "detail": "pushed to origin/main", "elapsed": elapsed}

    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start
        return {"label": label, "status": "FAILED",
                "detail": f"git command timed out after {GIT_TIMEOUT_SEC}s", "elapsed": elapsed}
    except Exception as exc:
        elapsed = time.monotonic() - start
        return {"label": label, "status": "FAILED", "detail": _redact(str(exc))[:200], "elapsed": elapsed}


STEP_PLAN = [
    dict(script_path=FETCH_X_SCRIPT, label="fetch_x_data", extra_args=["--recent"], timeout_sec=FETCH_TIMEOUT_SEC),
    dict(script_path=CLEAN_SCRIPT, label="clean_x_data"),
    dict(script_path=PREPROCESS_SCRIPT, label="preprocess_x_text"),
    dict(script_path=HASHTAGS_SCRIPT, label="extract_topics_hashtags"),
    dict(script_path=ACTOR_MENTIONS_SCRIPT, label="extract_actor_mentions"),
    dict(script_path=SENTIMENT_SCRIPT, label="add_sentiment"),
    dict(script_path=RELATIONS_SCRIPT, label="relevance_and_relations", extra_args=["--sample-size", "0"],
         timeout_sec=RELATIONS_TIMEOUT_SEC, quota_aware=True),
    dict(script_path=NETWORK_V2_SCRIPT, label="build_network_v2"),
    dict(script_path=TIMELINE_SCRIPT, label="build_daily_timelines", extra_args=["--x-only"]),
]


def main():
    run_start = time.monotonic()
    run_start_iso = _now_iso()
    results = []

    # Live per-step progress printing - added after a real incident where a
    # manual run sat with ZERO terminal output for 10+ minutes and looked
    # indistinguishable from "frozen/broken" to someone watching it, when it
    # was actually just deep inside one still-running step: subprocess.run()
    # in run_step() uses capture_output=True, so a step's own stdout/stderr
    # is invisible until THAT step's subprocess.run() call returns, and
    # write_status_log() below only writes pipeline_status.log ONCE, after
    # every single step in STEP_PLAN has finished - so previously there was
    # no way to tell "still working, be patient" apart from "actually
    # stuck" without waiting out the full run or a per-step timeout (up to
    # 45 minutes for relevance_and_relations). These two prints don't change
    # run_step()'s behavior, results shape, or pipeline_status.log's format
    # at all - purely additive, for whoever is watching this terminal live.
    # If a step DOES hang, whichever "... starting" line printed last with
    # no matching "->" line after it is the stuck one - that's now visible
    # immediately instead of only inferable after the fact.
    total_steps = len(STEP_PLAN)

    # Belt-and-suspenders: run_step() itself never raises, so this loop
    # should never actually need the except below - but the whole point of
    # this rewrite is that NOTHING is allowed to crash the run, including
    # a bug in the orchestrator's own loop. If a step somehow still throws,
    # record it as a FAILED result (not a bare traceback) and carry on to
    # the rest of the steps rather than aborting the whole pipeline.
    for i, step in enumerate(STEP_PLAN):
        label = step["label"]
        print(f"[{i + 1}/{total_steps}] {label} starting...", flush=True)
        try:
            result = run_step(**step)
        except Exception as exc:
            result = {"label": label, "status": "FAILED",
                      "detail": f"orchestrator error: {exc}"[:200], "elapsed": 0.0}
        results.append(result)
        detail_suffix = f" - {result['detail']}" if result["detail"] else ""
        print(f"    -> {result['status']} ({result['elapsed']:.0f}s){detail_suffix}", flush=True)
        # Any step not yet attempted (if the loop itself were somehow
        # interrupted) still gets logged as not-run, so the log's step
        # count always matches STEP_PLAN's length.

    # Runs last, and separately from the STEP_PLAN loop above (not via
    # run_step()) because it needs `results` itself as input for a richer
    # commit message. Same belt-and-suspenders wrapping as every other step:
    # sync_to_git() never raises on its own, but the loop's own robustness
    # philosophy applies here too - a bug in this call must never take down
    # the run or block the status log from being written.
    #
    # Gated behind ENABLE_GIT_SYNC (see that constant's comment above) -
    # while running local-only (no Streamlit Community Cloud deployment),
    # this step is skipped entirely and left out of results/the log, not
    # logged as SKIPPED/FAILED every cycle.
    if ENABLE_GIT_SYNC:
        print(f"[{total_steps + 1}/{total_steps + 1}] sync_to_github starting...", flush=True)
        try:
            result = sync_to_git(results)
        except Exception as exc:
            result = {"label": "sync_to_github", "status": "FAILED",
                      "detail": f"orchestrator error: {exc}"[:200], "elapsed": 0.0}
        results.append(result)
        detail_suffix = f" - {result['detail']}" if result["detail"] else ""
        print(f"    -> {result['status']} ({result['elapsed']:.0f}s){detail_suffix}", flush=True)

    total_elapsed = time.monotonic() - run_start
    write_status_log(results, run_start_iso, total_elapsed)

    print(f"\nPipeline run finished in {total_elapsed:.0f}s. See {os.path.basename(STATUS_LOG)} for details:")
    for r in results:
        print(f"  {r['label']:26s} {r['status']}" + (f" - {r['detail']}" if r["detail"] else ""))


if __name__ == "__main__":
    main()

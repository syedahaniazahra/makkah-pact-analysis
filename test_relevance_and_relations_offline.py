"""
Offline verification for relevance_and_relations.py's BATCHED prompt-building,
response-parsing, batch-sizing, and token-pacing logic, using REAL post text
pulled from the real x_data_cleaned.csv (the same 10 rows used to verify v1,
now exercised through the batched code paths) plus HAND-WRITTEN mock Groq
batch responses - including deliberately messy ones (a post_id the model
drops entirely, a hallucinated post_id, a hallucinated actor pair, an invalid
relation_type, a non-boolean topic_relevant, a totally-broken batch, markdown
fencing) so the parsing survives real model sloppiness at batch scale, not
just the single-post happy path v1 already covered.

TokenPacer is tested with an injected fake clock/sleep function, so its
60-second-window accounting is verified without an actual 60-second wait.

No network call is made anywhere in this file. Run with:
    python test_relevance_and_relations_offline.py
Exits non-zero (and prints which assertion failed) if anything is wrong.
"""

import json
import os
import sys
import tempfile

import pandas as pd

import relevance_and_relations as rar
from relevance_and_relations import (
    ABSOLUTE_MAX_COMPLETION_TOKENS,
    BATCH_SIZE_DEFAULT,
    MAX_BATCH_INPUT_TOKENS_DEFAULT,
    MAX_TOKENS_PER_REQUEST_HARD_CAP,
    OTPM_LIMIT,
    OTPM_SAFETY_FACTOR,
    PER_POST_COMPLETION_CEILING_FULL,
    PER_POST_COMPLETION_CEILING_RELATIONS,
    RELATION_TYPES,
    TPM_LIMIT,
    TokenPacer,
    call_groq_with_retry,
    build_batches,
    build_batch_prompt_body,
    build_full_batch_messages,
    build_relations_only_batch_messages,
    build_sample_idx,
    estimate_tokens,
    parse_actors,
    parse_batch_response,
    parse_rate_limit_hint,
    purge_relations_for_post_ids,
)

# Real rows from the real x_data_cleaned.csv (id, clean_text, actors_mentioned)
SAMPLE_ROWS = [
    {"id": "2091519808525259143",
     "text": "the mecca pact will include everyone (sunni nations) except iran and uae. after israel turkey will attack iran because they will attack azerbaijcan (2030's)",
     "actors_raw": "['Iran', 'Turkiye']"},
    {"id": "2091511994381685126",
     "text": ("iran invited to the mecca pact: what does it mean? iran confirmed it was invited to join the mecca pact, "
               "the defensive military alliance between turkey, saudi arabia, and pakistan. although no word has been "
               "said by the 3 founding members, the secrecy is strategic. officially confirming iran's entry before the "
               "terms regarding militias are negotiated would trigger immediate, harsh diplomatic retaliation from "
               "israel and the united states."),
     "actors_raw": "['Iran', 'Pakistan', 'Saudi Arabia', 'Turkiye', 'United States']"},
    {"id": "2091511523604857250",
     "text": "al-monitor turkey: bagging bangladesh would be quite a coup for the turkey-pakistan-saudi defense alliance - above all for pakistan, given its traditional rivalry with india, writes",
     "actors_raw": "['Pakistan', 'Saudi Arabia', 'Turkiye']"},
    {"id": "2091510877426241595",
     "text": "iran reportedly invited to join makkah defence pact amid saudi-houthi tensions read:",
     "actors_raw": "['Iran', 'Saudi Arabia']"},
    {"id": "2091509886840356891",
     "text": "file under wtf: saudi arabia, turkey, pakistan invited iran to join newly-established mecca defense pact",
     "actors_raw": "['Iran', 'Pakistan', 'Saudi Arabia', 'Turkiye']"},
    {"id": "2091518940409418032",
     "text": "if iran joins in, bangladesh & uae too become members then there are very few countries that the mecca defence pact can be operationalised against.",
     "actors_raw": "['Iran']"},
    {"id": "2091518640873185297",
     "text": "those bangladeshi radical jihadist razakars want to join mecca pact alongwith pakistan to win a war against india to take the revenge of their defeat in 1971.",
     "actors_raw": "['Pakistan']"},
    {"id": "2091518575366316345",
     "text": "it would be horrible mecca pact is comprised of us puppets iran is not a us puppet",
     "actors_raw": "['Iran']"},
    {"id": "2091518398807019644",
     "text": "is this and the iran invite to mecca pact real??",
     "actors_raw": "['Iran']"},
    {"id": "9999999999999999998",
     "text": "iran just won the world cup, incredible game last night",
     "actors_raw": "['Iran']"},
]

for row in SAMPLE_ROWS:
    row["actors"] = parse_actors(row["actors_raw"])

failures = []


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        failures.append(label)


# --------------------------------------------------------------------------
# 1. estimate_tokens sanity
# --------------------------------------------------------------------------
print("=== estimate_tokens ===")
check("empty string -> at least 1", estimate_tokens("") >= 1)
check("longer text -> more estimated tokens", estimate_tokens("a" * 400) > estimate_tokens("a" * 40))

# --------------------------------------------------------------------------
# 2. build_batches - respects BOTH the row-count cap and the token cap
# --------------------------------------------------------------------------
print("\n=== build_batches ===")
idx_list = list(range(len(SAMPLE_ROWS)))


class FakeDF:
    """Minimal stand-in so build_batches can call df.at[idx, 'clean_text']
    against our SAMPLE_ROWS list by position, without needing real pandas
    row indices."""
    def __init__(self, rows):
        self.rows = rows

    class _At:
        def __init__(self, rows):
            self.rows = rows

        def __getitem__(self, key):
            idx, col = key
            return self.rows[idx]["text"] if col == "clean_text" else None

    @property
    def at(self):
        return FakeDF._At(self.rows)


fake_df = FakeDF(SAMPLE_ROWS)

batches_by_count = build_batches(idx_list, fake_df, max_rows_per_batch=4, max_input_tokens_per_batch=100000)
check("row-count cap: 10 posts / max 4 per batch -> 3 batches", len(batches_by_count) == 3)
check("row-count cap: batch sizes are 4, 4, 2", [len(b) for b in batches_by_count] == [4, 4, 2])
check("row-count cap: every index appears exactly once, none dropped/duplicated",
      sorted(sum(batches_by_count, [])) == sorted(idx_list))

# SAMPLE_ROWS[1] is the long multi-paragraph post - a tiny token cap should
# force it into its own batch (or a very small one) even though rows are
# otherwise cheap.
batches_by_tokens = build_batches(idx_list, fake_df, max_rows_per_batch=10, max_input_tokens_per_batch=60)
check("token cap: the long post doesn't get bundled with everything else",
      any(len(b) == 1 and b[0] == 1 for b in batches_by_tokens) or
      all(len(b) < 10 for b in batches_by_tokens))
check("token cap: every index still appears exactly once",
      sorted(sum(batches_by_tokens, [])) == sorted(idx_list))
check("token cap: no batch is empty", all(len(b) > 0 for b in batches_by_tokens))

# --------------------------------------------------------------------------
# 3. Prompt builders
# --------------------------------------------------------------------------
print("\n=== prompt builders ===")
batch = [{"post_id": r["id"], "text": r["text"], "actors": r["actors"]} for r in SAMPLE_ROWS[:4]]
body = build_batch_prompt_body(batch)
check("prompt body includes every post_id in the batch", all(r["post_id"] in body for r in batch))
check("prompt body includes every post's text", all(r["text"] in body for r in batch))

full_msgs = build_full_batch_messages(batch)
check("full batch messages: system+user roles", [m["role"] for m in full_msgs] == ["system", "user"])
check("full batch system prompt mentions topic_relevant", "topic_relevant" in full_msgs[0]["content"])

rel_msgs = build_relations_only_batch_messages(batch)
check("relations-only batch messages: system+user roles", [m["role"] for m in rel_msgs] == ["system", "user"])
check("relations-only system prompt does NOT ask for topic_relevant (already known)",
      "topic_relevant" not in rel_msgs[0]["content"])

# --------------------------------------------------------------------------
# 4. parse_batch_response - mode="full", clean batch of 4
# --------------------------------------------------------------------------
print("\n=== parse_batch_response: mode=full, clean batch ===")
batch4 = SAMPLE_ROWS[:4]  # ids: [...143, ...126, ...250, ...595]
expected4 = {r["id"]: {"actors": r["actors"]} for r in batch4}

def rel(a1, a2, rtype):
    return {"actor_1": a1, "actor_2": a2, "relation_type": rtype}


# Built via json.dumps of real Python structures, not hand-typed brace
# counting - a hand-typed batch JSON string is exactly the kind of thing
# that's easy to get subtly wrong (extra/missing brace) without it being
# obvious from reading it, which is precisely the failure mode this test
# exists to catch in the SCRIPT's own parsing, not reproduce in the test.
clean_batch = {
    "results": [
        {"post_id": batch4[0]["id"], "topic_relevant": True, "relations": [rel("Iran", "Turkiye", "hostile")]},
        {"post_id": batch4[1]["id"], "topic_relevant": True,
         "relations": [rel("Iran", "Pakistan", "neutral-reporting"), rel("Iran", "Saudi Arabia", "mediating")]},
        {"post_id": batch4[2]["id"], "topic_relevant": True, "relations": []},
        {"post_id": batch4[3]["id"], "topic_relevant": False, "relations": []},
    ]
}
clean_batch_json = json.dumps(clean_batch)
results, err = parse_batch_response(clean_batch_json, expected4, "full")
check("clean batch: no batch-level error", err is None)
check("clean batch: all 4 post_ids present in results", set(results.keys()) == set(expected4.keys()))
check("clean batch: post 0 relevant=True with 1 relation",
      results[batch4[0]["id"]]["topic_relevant"] is True and results[batch4[0]["id"]]["relations"] == [("Iran", "Turkiye", "hostile")])
check("clean batch: post 1 relevant=True with 2 relations",
      len(results[batch4[1]["id"]]["relations"]) == 2)
check("clean batch: post 3 relevant=False", results[batch4[3]["id"]]["topic_relevant"] is False)
check("clean batch: no errors on any post", all(v["error"] is None for v in results.values()))

# --------------------------------------------------------------------------
# 5. Messiness: markdown fences, missing post, hallucinated post, bad actor,
#    bad relation_type, self-pair, non-boolean topic_relevant
# --------------------------------------------------------------------------
print("\n=== parse_batch_response: messy real-world responses ===")

fenced = f"Sure, here you go:\n```json\n{clean_batch_json}\n```"
results_f, err_f = parse_batch_response(fenced, expected4, "full")
check("markdown-fenced batch JSON still parses via regex fallback", err_f is None and len(results_f) == 4)

missing_one = json.dumps({
    "results": [dict(item) for item in clean_batch["results"][:3]]  # batch4[3] silently dropped by the model
})
results_m, err_m = parse_batch_response(missing_one, expected4, "full")
check("batch missing one post_id: no batch-level error (partial success allowed)", err_m is None)
check("batch missing one post_id: the missing one is flagged for retry, not silently lost",
      results_m[batch4[3]["id"]]["error"] == "missing from response")
check("batch missing one post_id: the other 3 still processed normally",
      all(results_m[batch4[i]["id"]]["error"] is None for i in (0, 1, 2)))

hallucinated_post = json.dumps({
    "results": [dict(item) for item in clean_batch["results"]] +
               [{"post_id": "0000000000000000000", "topic_relevant": True, "relations": []}]  # not in this batch
})
results_h, err_h = parse_batch_response(hallucinated_post, expected4, "full")
check("hallucinated extra post_id is ignored, not crashed on", err_h is None and "0000000000000000000" not in results_h)

hallucinated_actor = json.dumps({
    "results": [
        {"post_id": batch4[0]["id"], "topic_relevant": True,
         "relations": [rel("Iran", "Israel", "hostile")]},  # Israel not a listed actor for this post
        {"post_id": batch4[1]["id"], "topic_relevant": True, "relations": []},
        {"post_id": batch4[2]["id"], "topic_relevant": True, "relations": []},
        {"post_id": batch4[3]["id"], "topic_relevant": False, "relations": []},
    ]
})
results_ha, err_ha = parse_batch_response(hallucinated_actor, expected4, "full")
check("hallucinated actor pair silently dropped, post still succeeds",
      err_ha is None and results_ha[batch4[0]["id"]]["relations"] == [] and results_ha[batch4[0]["id"]]["error"] is None)

bad_type_and_self_pair = json.dumps({
    "results": [
        {"post_id": batch4[0]["id"], "topic_relevant": True,
         "relations": [rel("Iran", "Turkiye", "best friends"), rel("Iran", "Iran", "hostile")]},
        {"post_id": batch4[1]["id"], "topic_relevant": True, "relations": []},
        {"post_id": batch4[2]["id"], "topic_relevant": True, "relations": []},
        {"post_id": batch4[3]["id"], "topic_relevant": False, "relations": []},
    ]
})
results_bt, err_bt = parse_batch_response(bad_type_and_self_pair, expected4, "full")
check("invalid relation_type falls back to 'unclear'", results_bt[batch4[0]["id"]]["relations"] == [("Iran", "Turkiye", "unclear")])
check("self-pair (actor_1 == actor_2) is dropped, not included",
      ("Iran", "Iran", "hostile") not in results_bt[batch4[0]["id"]]["relations"])

non_bool_relevant = json.dumps({
    "results": [
        {"post_id": batch4[0]["id"], "topic_relevant": "yes", "relations": []},  # string, not bool
        {"post_id": batch4[1]["id"], "topic_relevant": True, "relations": []},
        {"post_id": batch4[2]["id"], "topic_relevant": True, "relations": []},
        {"post_id": batch4[3]["id"], "topic_relevant": False, "relations": []},
    ]
})
results_nb, err_nb = parse_batch_response(non_bool_relevant, expected4, "full")
check("non-boolean topic_relevant flagged as an error for that post only",
      results_nb[batch4[0]["id"]]["error"] == "topic_relevant missing or not boolean")
check("...but the other 3 posts in the same batch are unaffected",
      all(results_nb[batch4[i]["id"]]["error"] is None for i in (1, 2, 3)))

totally_broken = "I'm not able to provide that in JSON format, sorry!"
results_tb, err_tb = parse_batch_response(totally_broken, expected4, "full")
check("totally non-JSON reply: whole batch flagged as a batch-level error", err_tb is not None and results_tb == {})

# --------------------------------------------------------------------------
# 6. mode="relations_only" - topic_relevant is assumed True, not asked for
# --------------------------------------------------------------------------
print("\n=== parse_batch_response: mode=relations_only ===")
batch2 = SAMPLE_ROWS[4:6]
expected2 = {r["id"]: {"actors": r["actors"]} for r in batch2}
all_pairs = [
    rel("Iran", "Pakistan", "supportive"), rel("Iran", "Saudi Arabia", "supportive"),
    rel("Iran", "Turkiye", "supportive"), rel("Pakistan", "Saudi Arabia", "supportive"),
    rel("Pakistan", "Turkiye", "supportive"), rel("Saudi Arabia", "Turkiye", "supportive"),
]
relations_only_json = json.dumps({
    "results": [
        {"post_id": batch2[0]["id"], "relations": all_pairs},
        {"post_id": batch2[1]["id"], "relations": []},
    ]
})
results_ro, err_ro = parse_batch_response(relations_only_json, expected2, "relations_only")
check("relations_only: no batch error", err_ro is None)
check("relations_only: topic_relevant assumed True without being asked", results_ro[batch2[0]["id"]]["topic_relevant"] is True)
check("relations_only: 4-actor post gets all 6 pairs", len(results_ro[batch2[0]["id"]]["relations"]) == 6)

# --------------------------------------------------------------------------
# 7. TokenPacer - fake clock/sleep, no real waiting
# --------------------------------------------------------------------------
print("\n=== TokenPacer (simulated time) ===")


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


clock = FakeClock()
sleep_calls = []


def fake_sleep(seconds):
    sleep_calls.append(seconds)
    clock.advance(seconds)


pacer = TokenPacer(tpm_limit=1000, safety_factor=1.0, clock=clock, sleep_fn=fake_sleep)  # budget = 1000

pacer.record(600)
waited = pacer.wait_for_budget(300)  # 600+300=900 <= 1000
check("under budget: wait_for_budget returns without sleeping", waited is False and len(sleep_calls) == 0)
check("used_last_minute reflects the recorded call", pacer.used_last_minute() == 600)

pacer.record(300)  # now 900 used
check("used_last_minute after second record", pacer.used_last_minute() == 900)

waited2 = pacer.wait_for_budget(200)  # 900+200=1100 > 1000 -> must wait for the window to age out
check("over budget: wait_for_budget actually slept", waited2 is True and len(sleep_calls) > 0)
check("over budget: simulated clock advanced (proves it 'waited' without a real sleep)", clock.t > 0)
check("over budget: after waiting long enough, usage is back under budget", pacer.used_last_minute() + 200 <= 1000)

# --------------------------------------------------------------------------
# 8. parse_rate_limit_hint - including the EXACT real error message Groq
#    returned during actual testing (a 200,000/day TPD wall)
# --------------------------------------------------------------------------
print("\n=== parse_rate_limit_hint ===")

real_tpd_message = (
    "Error code: 429 - {'error': {'message': 'Rate limit reached for model "
    "`openai/gpt-oss-20b` in organization `org_01jyvy0z4ce5tspfm0sbqhh2z8` "
    "service tier `on_demand` on tokens per day (TPD): Limit 200000, Used "
    "197088, Requested 4151. Please try again in 8m55.248s. Need more tokens? "
    "Upgrade to Dev Tier today at https://console.groq.com/settings/billing', "
    "'type': 'tokens', 'code': 'rate_limit_exceeded'}}"
)
limit_type, wait = parse_rate_limit_hint(real_tpd_message)
check("real TPD message: limit type correctly identified as TPD", limit_type == "TPD")
check("real TPD message: wait parsed as 8m55.248s = 535.248s",
      wait is not None and abs(wait - 535.248) < 0.01)
check("real TPD message: this wait exceeds the 45s retry cap (would correctly fail fast, not loop)",
      wait > 45)

short_tpm_message = (
    "Rate limit reached for model `openai/gpt-oss-20b` on tokens per minute (TPM): "
    "Limit 8000, Used 7800, Requested 500. Please try again in 3.2s."
)
limit_type2, wait2 = parse_rate_limit_hint(short_tpm_message)
check("short TPM message: limit type correctly identified as TPM", limit_type2 == "TPM")
check("short TPM message: wait parsed as ~3.2s", wait2 is not None and abs(wait2 - 3.2) < 0.01)
check("short TPM message: this wait is under the 45s cap (would correctly keep retrying)", wait2 <= 45)

rpd_message = "on requests per day (RPD): Limit 1000, Used 1000, Requested 1. Please try again in 2h14m0s."
limit_type3, wait3 = parse_rate_limit_hint(rpd_message)
check("RPD message: limit type correctly identified as RPD", limit_type3 == "RPD")
check("RPD message: wait parsed as 2h14m = 8040s", wait3 is not None and abs(wait3 - 8040.0) < 0.01)

unparseable_message = "Something went wrong, try again later."
limit_type4, wait4 = parse_rate_limit_hint(unparseable_message)
check("unparseable message: limit type is None, not a crash", limit_type4 is None)
check("unparseable message: wait is None, not a crash", wait4 is None)

# --------------------------------------------------------------------------
# 9. build_sample_idx - engagement-based sampling that never discards
#    already-done rows, regardless of --sample-size
# --------------------------------------------------------------------------
print("\n=== build_sample_idx ===")

sample_df = pd.DataFrame({
    "detected_language": ["en"] * 10 + ["fr", "fr"],
    "relations_checked": pd.array([True] * 5 + [pd.NA] * 5 + [pd.NA, pd.NA], dtype="boolean"),
    "likes":    [10, 20, 30, 40, 50,   5, 100, 15, 200,   1,  999, 999],
    "retweets": [1,   2,  3,  4,  5,   1,  50,  2,  20,   0,  999, 999],
})
# rows 0-4: already done (engagement 11,22,33,44,55) - must always survive
# rows 5-9: not done (engagement  6,150, 17,220, 1)  - candidates for sampling
# rows 10-11: non-English, huge engagement - must NEVER appear regardless of sample_size
en_mask_s = sample_df["detected_language"] == "en"

in_scope0, done0, new0 = build_sample_idx(sample_df, en_mask_s, 0)
check("sample_size=0: already_done_count is 5", done0 == 5)
check("sample_size=0 (no cap): every not-done English row included", new0 == 5)
check("sample_size=0: in_scope is exactly the 10 English rows, no non-English", in_scope0 == set(range(10)))

# 5 already done + 2 more needed -> top 2 not-done by engagement: idx 8 (220), idx 6 (150)
in_scope2, done2, new2 = build_sample_idx(sample_df, en_mask_s, 7)
check("sample_size=7: already_done_count still 5", done2 == 5)
check("sample_size=7: exactly 2 newly sampled (to reach target of 7)", new2 == 2)
check("sample_size=7: the 2 highest-engagement not-done rows are chosen (idx 8 and 6, not idx 7/5/9)",
      in_scope2 == set(range(5)) | {8, 6})

# sample_size BELOW the already-done count - must not trim already-done work
in_scope3, done3, new3 = build_sample_idx(sample_df, en_mask_s, 3)
check("sample_size=3 (below already-done count of 5): zero new rows added", new3 == 0)
check("sample_size=3: all 5 already-done rows are STILL kept, not discarded down to 3",
      in_scope3 == set(range(5)))

check("non-English rows never included regardless of sample_size",
      10 not in in_scope0 and 11 not in in_scope0 and 10 not in in_scope2 and 11 not in in_scope2)

nan_df = sample_df.copy()
nan_df.loc[6, "likes"] = float("nan")  # idx6: likes NaN, retweets 50 -> engagement should coerce to 0+50=50
in_scope_nan, _, new_nan = build_sample_idx(nan_df, en_mask_s, 7)
check("NaN likes value coerced to 0 instead of crashing the sort", new_nan == 2)
check("NaN-likes row still ranks correctly on its remaining (retweets) value",
      in_scope_nan == set(range(5)) | {8, 6})

# --------------------------------------------------------------------------
# 10. v2.3 pair-connection tightening - mocked against the 10 real diagnostic
#     examples (the "unclear" bucket spot-check from the conversation)
# --------------------------------------------------------------------------
# No network call is made here either - this cannot test the MODEL's actual
# judgment on these posts (only a real Groq call under the new prompt can do
# that). What it verifies is that the pipeline correctly represents all three
# possible outcomes end-to-end once the model does answer: a SKIPPED pair
# (the model, per the new instructions, emits no relation entry at all)
# parses as an empty relations list rather than an error or a phantom
# 'unclear'; a genuinely-hedged pair correctly parses as a kept 'unclear'
# entry; and a kept judgment-call entry with a real (non-'unclear') type also
# parses cleanly. The MOCK responses below encode the EXPECTED outcome per
# example per the manual diagnostic read already done - ex4/ex6/ex8/ex10 as
# "skip" (post never connects that specific pair), ex1/ex5 as "unclear"
# (genuinely open questions), ex3/ex9 as "judgment" (defensible either way -
# mocked here as kept, to prove a kept judgment-call entry also parses fine).
print("\n=== v2.3 pair-connection tightening (10-example diagnostic mock) ===")

DIAGNOSTIC_EXAMPLES = [
    # (post_id, actor_1, actor_2, expected_outcome) - expected_outcome is one of
    # "skip" (no relation row expected), "unclear" (kept, genuinely hedged), or
    # "judgment" (defensible either way - mocked as kept here)
    ("ex1", "Iran", "Saudi Arabia", "unclear"),       # "How does Iran view the pact?" - open question
    ("ex2", "Saudi Arabia", "Turkiye", "skip"),       # Greece-focused article, no direct link stated
    ("ex3", "Saudi Arabia", "Turkiye", "judgment"),   # pact co-membership implied, not stated for this pair
    ("ex4", "Pakistan", "United States", "skip"),     # each named separately, never connected to each other
    ("ex5", "Saudi Arabia", "Turkiye", "unclear"),    # "will Saudi invoke the pact...?" - speculative question
    ("ex6", "Iran", "Saudi Arabia", "skip"),          # co-mentioned in a broad roundup, no link stated
    ("ex7", "Iran", "Pakistan", "skip"),              # both named in analytical piece, no link stated
    ("ex8", "Saudi Arabia", "Turkiye", "skip"),       # Bangladesh-interest post, no direct link stated
    ("ex9", "Turkiye", "United States", "judgment"),  # "could complicate U.S. plans" - soft implied link
    ("ex10", "Iran", "Saudi Arabia", "skip"),         # both named in a list of actors, no link stated
]

expected = {
    post_id: {"actors": [a1, a2]}
    for post_id, a1, a2, _ in DIAGNOSTIC_EXAMPLES
}

mock_results = []
for post_id, a1, a2, outcome in DIAGNOSTIC_EXAMPLES:
    if outcome == "unclear":
        relations = [{"actor_1": a1, "actor_2": a2, "relation_type": "unclear"}]
    elif outcome == "judgment":
        relations = [{"actor_1": a1, "actor_2": a2, "relation_type": "neutral-reporting"}]
    else:  # "skip" - the tightened prompt should omit the pair, not emit anything for it
        relations = []
    mock_results.append({"post_id": post_id, "relations": relations})

mock_raw = json.dumps({"results": mock_results})
diag_results, diag_batch_err = parse_batch_response(mock_raw, expected, mode="relations_only")

check("v2.3 mock batch: no batch-level parse error", diag_batch_err is None)
for post_id, a1, a2, outcome in DIAGNOSTIC_EXAMPLES:
    res = diag_results.get(post_id)
    check(f"v2.3 mock: {post_id} parsed without a per-row error", res is not None and res["error"] is None)
    rels = res["relations"] if res else None
    if outcome == "skip":
        check(f"v2.3 mock: {post_id} ({a1}/{a2}) correctly SKIPPED - empty relations, not a phantom 'unclear'",
              rels == [])
    elif outcome == "unclear":
        check(f"v2.3 mock: {post_id} ({a1}/{a2}) correctly KEPT as 'unclear' (genuinely hedged)",
              rels == [(a1, a2, "unclear")])
    else:  # judgment
        check(f"v2.3 mock: {post_id} ({a1}/{a2}) kept judgment-call entry parses cleanly",
              rels == [(a1, a2, "neutral-reporting")])

print(
    "[v2.3 diagnostic summary] ex2/ex4/ex6/ex7/ex8/ex10 expected to be SKIPPED entirely once the real "
    "recheck runs (text never connects that specific pair); ex1/ex5 expected to correctly remain "
    "'unclear' (genuinely open/hedged questions); ex3/ex9 are judgment calls the tightened prompt "
    "could reasonably decide either way - worth a manual look once the real recheck completes. This "
    "offline pass proves the parsing pipeline handles all three outcomes correctly; it cannot test "
    "the model's actual judgment without a live API call."
)

# --------------------------------------------------------------------------
# 11. purge_relations_for_post_ids - used by --recheck-relations to drop
#     stale relation rows before re-extracting under the new prompt
# --------------------------------------------------------------------------
print("\n=== purge_relations_for_post_ids ===")

with tempfile.TemporaryDirectory() as tmp_dir:
    tmp_relations_csv = os.path.join(tmp_dir, "actor_relations.csv")
    original_relations_csv = rar.RELATIONS_CSV
    rar.RELATIONS_CSV = tmp_relations_csv  # monkeypatch the module-level path for this test only
    try:
        with open(tmp_relations_csv, "w", newline="", encoding="utf-8") as f:
            f.write("post_id,actor_1,actor_2,relation_type\n")
            f.write("100,Iran,Saudi Arabia,unclear\n")
            f.write("101,Pakistan,Turkiye,supportive\n")
            f.write("100,Iran,Pakistan,hostile\n")   # second row for post 100 - both should be purged
            f.write("102,Saudi Arabia,Turkiye,neutral-reporting\n")

        removed = purge_relations_for_post_ids(["100"])
        check("purge: removes both rows for a targeted post_id (2 rows for post 100)", removed == 2)

        with open(tmp_relations_csv, "r", encoding="utf-8") as f:
            remaining_lines = f.read().splitlines()
        check("purge: header preserved", remaining_lines[0] == "post_id,actor_1,actor_2,relation_type")
        check("purge: untouched post_ids (101, 102) still present, post 100 fully gone",
              remaining_lines[1:] == ["101,Pakistan,Turkiye,supportive",
                                       "102,Saudi Arabia,Turkiye,neutral-reporting"])

        removed_none = purge_relations_for_post_ids(["999"])
        check("purge: targeting a post_id with no rows removes 0 and doesn't error", removed_none == 0)

        removed_second_call = purge_relations_for_post_ids(["101"])  # file still exists here, sanity check only
        check("purge: still works normally on a second call", removed_second_call == 1)
    finally:
        rar.RELATIONS_CSV = original_relations_csv  # restore, even if a check above failed

    # No-file case, in a fresh temp dir where the file was never created
    fresh_tmp_dir = tempfile.mkdtemp()
    rar.RELATIONS_CSV = os.path.join(fresh_tmp_dir, "does_not_exist.csv")
    try:
        removed_no_file = purge_relations_for_post_ids(["100"])
        check("purge: no-op (returns 0) when RELATIONS_CSV doesn't exist yet", removed_no_file == 0)
    finally:
        rar.RELATIONS_CSV = original_relations_csv

# --------------------------------------------------------------------------
# 12. recheck-relations target selection (v2.3.1 FIXED logic) - selects
#     purely on relations_checked, deliberately ignoring
#     relations_recheck_done entirely (that's the fix - see the v2.3.1
#     docstring section and run_recheck_relations for the incident this
#     replaced: relations_recheck_done being marked True upfront, before
#     work was confirmed, made a real --recheck-relations run falsely
#     report "nothing left to recheck" after being interrupted by a rate
#     wall, even though most of the target set was never reprocessed)
# --------------------------------------------------------------------------
print("\n=== recheck-relations target selection (v2.3.1 fix) ===")

recheck_df = pd.DataFrame({
    "id": ["a", "b", "c", "d", "e", "f", "g"],
    "detected_language": ["en", "en", "en", "en", "en", "fr", "en"],
    "topic_relevant": pd.array([True, True, False, True, pd.NA, True, True], dtype="boolean"),
    "relations_checked": pd.array([pd.NA, True, True, pd.NA, pd.NA, True, pd.NA], dtype="boolean"),
    # row g deliberately has relations_recheck_done=True (mimicking the REAL, currently-broken
    # state on disk: wrongly marked True upfront) while relations_checked is still NA - the fix
    # must target it anyway, proving the broken column is never consulted any more.
    "relations_recheck_done": pd.array([pd.NA, True, pd.NA, pd.NA, pd.NA, pd.NA, True], dtype="boolean"),
})
recheck_en_mask = recheck_df["detected_language"] == "en"

# This is the EXACT expression run_recheck_relations() now uses - relations_recheck_done is not
# referenced anywhere in it.
target_mask = recheck_en_mask & (recheck_df["topic_relevant"].fillna(False) == True) & recheck_df["relations_checked"].isna()  # noqa: E712
target_ids = set(recheck_df.loc[target_mask, "id"])

# row a: en, topic_relevant=True, relations_checked NA -> TARGETED
# row b: en, topic_relevant=True, relations_checked=True (genuinely done) -> NOT targeted
# row c: en, topic_relevant=False -> never had relations, not targeted
# row d: en, topic_relevant=True, relations_checked still NA (never finished) -> TARGETED
# row e: en, topic_relevant is NA (never even relevance-checked) -> not targeted
# row f: NOT English -> never targeted regardless of other columns
# row g: en, topic_relevant=True, relations_checked NA, but relations_recheck_done WRONGLY True
#        (mimics the real broken state on disk) -> MUST still be targeted - proves the fix
check("recheck target selection: row a (outstanding) is targeted", "a" in target_ids)
check("recheck target selection: row b (relations_checked=True, genuinely done) is NOT targeted", "b" not in target_ids)
check("recheck target selection: row c (topic_relevant=False) is not targeted", "c" not in target_ids)
check("recheck target selection: row d (never finished relations) is still targeted", "d" in target_ids)
check("recheck target selection: row e (topic_relevant never set) is not targeted", "e" not in target_ids)
check("recheck target selection: row f (non-English) is never targeted", "f" not in target_ids)
check("recheck target selection: row g (outstanding despite a WRONG relations_recheck_done=True, "
      "matching the real broken state on disk) is STILL targeted - the fix", "g" in target_ids)
check("recheck target selection: exactly {a, d, g}", target_ids == {"a", "d", "g"})

# --------------------------------------------------------------------------
# 13. process_group's purge_before_batch / recheck_marker_col (v2.3.1) -
#     confirms purging happens per-batch, only for CONFIRMED results, and
#     never for a batch that fails to parse; and that recheck_marker_col is
#     set only alongside a confirmed relations_checked, never upfront.
# --------------------------------------------------------------------------
print("\n=== process_group: purge_before_batch / recheck_marker_col ===")


class _FakeArgs:
    batch_size = 7
    max_batch_input_tokens = 2000


class _FakePacer:
    def wait_for_budget(self, *a, **kw):
        return False

    def record(self, *a, **kw):
        pass


def _fake_client(responses_by_call):
    """responses_by_call: list of raw JSON strings, one per expected call,
    returned in order. Mimics just enough of the Groq SDK response shape
    for process_group/call_groq_with_retry to work with (usage.total_tokens
    and choices[0].message.content)."""
    calls = {"n": 0}

    class _Msg:
        def __init__(self, content):
            self.content = content
            self.reasoning = None

    class _Choice:
        def __init__(self, content):
            self.message = _Msg(content)
            self.finish_reason = "stop"

    class _Usage:
        total_tokens = 50

    class _Resp:
        def __init__(self, content):
            self.choices = [_Choice(content)]
            self.usage = _Usage()

    class _Completions:
        def create(self, **kwargs):
            content = responses_by_call[calls["n"]]
            calls["n"] += 1
            return _Resp(content)

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    return _Client()


with tempfile.TemporaryDirectory() as tmp_dir:
    tmp_relations_csv = os.path.join(tmp_dir, "actor_relations.csv")
    original_relations_csv = rar.RELATIONS_CSV
    rar.RELATIONS_CSV = tmp_relations_csv
    try:
        # Pre-existing stale relations for posts 1 and 2 (as if from the old, over-permissive prompt).
        with open(tmp_relations_csv, "w", newline="", encoding="utf-8") as f:
            f.write("post_id,actor_1,actor_2,relation_type\n")
            f.write("1,Iran,Saudi Arabia,unclear\n")
            f.write("2,Pakistan,Turkiye,supportive\n")

        pg_df = pd.DataFrame({
            "id": ["1", "2"],
            "clean_text": ["post one text", "post two text"],
            "actors_mentioned": ["['Iran', 'Saudi Arabia']", "['Pakistan', 'Turkiye']"],
            "topic_relevant": pd.array([True, True], dtype="boolean"),
            "relations_checked": pd.array([pd.NA, pd.NA], dtype="boolean"),
            "relations_recheck_done": pd.array([pd.NA, pd.NA], dtype="boolean"),
        })

        # One batch (both rows fit under batch_size=7), model confirms post 1 with a NEW relation
        # and correctly omits post 2 entirely (simulating the tightened prompt dropping a pair).
        mock_response = json.dumps({"results": [
            {"post_id": "1", "relations": [{"actor_1": "Iran", "actor_2": "Saudi Arabia", "relation_type": "unclear"}]},
            {"post_id": "2", "relations": []},
        ]})
        client = _fake_client([mock_response])
        pg_counters = {"n_calls": 0, "n_rows_done": 0, "t_start": 0.0}
        rar.process_group(client, pg_df, [0, 1], "relations_only", _FakeArgs(), _FakePacer(), pg_counters,
                           purge_before_batch=True, recheck_marker_col="relations_recheck_done")

        check("purge_before_batch: both rows confirmed and counted", pg_counters["n_rows_done"] == 2)
        check("purge_before_batch: relations_checked set True for both confirmed rows",
              bool(pg_df.at[0, "relations_checked"]) and bool(pg_df.at[1, "relations_checked"]))
        check("recheck_marker_col: relations_recheck_done set True alongside relations_checked",
              bool(pg_df.at[0, "relations_recheck_done"]) and bool(pg_df.at[1, "relations_recheck_done"]))

        with open(tmp_relations_csv, "r", encoding="utf-8") as f:
            final_lines = f.read().splitlines()
        check("purge_before_batch: old stale row for post 1 replaced by the fresh one (not duplicated)",
              final_lines.count("1,Iran,Saudi Arabia,unclear") == 1)
        check("purge_before_batch: old stale row for post 2 purged, and NOT replaced (model omitted it)",
              "2,Pakistan,Turkiye,supportive" not in final_lines)
        check("purge_before_batch: no leftover relation row for post 2 at all",
              not any(line.startswith("2,") for line in final_lines))
    finally:
        rar.RELATIONS_CSV = original_relations_csv

    # Second scenario: a batch that fails to parse entirely must NOT purge anything for its posts.
    tmp_relations_csv_2 = os.path.join(tmp_dir, "actor_relations_2.csv")
    rar.RELATIONS_CSV = tmp_relations_csv_2
    try:
        with open(tmp_relations_csv_2, "w", newline="", encoding="utf-8") as f:
            f.write("post_id,actor_1,actor_2,relation_type\n")
            f.write("9,Iran,Pakistan,hostile\n")

        pg_df_2 = pd.DataFrame({
            "id": ["9"],
            "clean_text": ["post nine text"],
            "actors_mentioned": ["['Iran', 'Pakistan']"],
            "topic_relevant": pd.array([True], dtype="boolean"),
            "relations_checked": pd.array([pd.NA], dtype="boolean"),
            "relations_recheck_done": pd.array([pd.NA], dtype="boolean"),
        })
        client_2 = _fake_client(["this is not valid json at all"])
        pg_counters_2 = {"n_calls": 0, "n_rows_done": 0, "t_start": 0.0}
        rar.process_group(client_2, pg_df_2, [0], "relations_only", _FakeArgs(), _FakePacer(), pg_counters_2,
                           purge_before_batch=True, recheck_marker_col="relations_recheck_done")

        check("purge_before_batch: a batch that fails to parse leaves relations_checked untouched (still NA)",
              pd.isna(pg_df_2.at[0, "relations_checked"]))
        check("purge_before_batch: a batch that fails to parse does NOT purge that post's existing relations",
              open(tmp_relations_csv_2, encoding="utf-8").read().splitlines() ==
              ["post_id,actor_1,actor_2,relation_type", "9,Iran,Pakistan,hostile"])
    finally:
        rar.RELATIONS_CSV = original_relations_csv

# --------------------------------------------------------------------------
# 14. v2.4 OTPM-aware pacing - the SECOND, independent TokenPacer instance
#     paced against OTPM_LIMIT and fed only output (completion) tokens, plus
#     call_groq_with_retry actually waiting on / recording into it.
# --------------------------------------------------------------------------
print("\n=== v2.4 OTPM-aware pacing ===")

check("v2.5 sanity: BATCH_SIZE_DEFAULT reduced again, 4 to 3", BATCH_SIZE_DEFAULT == 3)
check("v2.4 sanity: OTPM_LIMIT is 1,000/min (from this account's real 429, not a public source)",
      OTPM_LIMIT == 1000)

# 14a. TokenPacer reused as-is for OTPM - same class, different budget/feed, exactly the
# claim made in the docstring ("TokenPacer itself needed no changes").
otpm_clock = FakeClock()
otpm_sleep_calls = []


def otpm_fake_sleep(seconds):
    otpm_sleep_calls.append(seconds)
    otpm_clock.advance(seconds)


otpm_test_pacer = TokenPacer(OTPM_LIMIT, safety_factor=OTPM_SAFETY_FACTOR, clock=otpm_clock, sleep_fn=otpm_fake_sleep)
check("OTPM pacer: budget is OTPM_LIMIT * OTPM_SAFETY_FACTOR = 850", otpm_test_pacer.budget == 850)

otpm_test_pacer.record(700)
waited_a = otpm_test_pacer.wait_for_budget(100)  # 700+100=800 <= 850
check("OTPM pacer: under its 850 budget, no wait", waited_a is False and len(otpm_sleep_calls) == 0)

otpm_test_pacer.record(100)  # now 800 used
waited_b = otpm_test_pacer.wait_for_budget(200)  # 800+200=1000 > 850 -> must wait
check("OTPM pacer: over its 850 budget, correctly waits (separately from any TPM pacer)",
      waited_b is True and len(otpm_sleep_calls) > 0)
check("OTPM pacer: after waiting, usage is back under its own budget",
      otpm_test_pacer.used_last_minute() + 200 <= 850)

# 14b. call_groq_with_retry: waits on BOTH pacers, and records ONLY completion_tokens
# (not total_tokens) into otpm_pacer - proving OTPM tracks output, not combined, usage.
shared_clock = FakeClock()
shared_sleep_calls = []


def shared_fake_sleep(seconds):
    shared_sleep_calls.append(seconds)
    shared_clock.advance(seconds)


# TPM pacer: generous budget, essentially never the bottleneck in this test.
cg_tpm_pacer = TokenPacer(100000, safety_factor=1.0, clock=shared_clock, sleep_fn=shared_fake_sleep)
# OTPM pacer: tight budget (850 after the 0.85 safety factor) - this is the one meant to bind.
cg_otpm_pacer = TokenPacer(OTPM_LIMIT, safety_factor=OTPM_SAFETY_FACTOR, clock=shared_clock, sleep_fn=shared_fake_sleep)


class _CGFakeUsage:
    def __init__(self, total, completion):
        self.total_tokens = total
        self.completion_tokens = completion


class _CGFakeMessage:
    def __init__(self, content):
        self.content = content
        self.reasoning = None


class _CGFakeChoice:
    def __init__(self, content):
        self.message = _CGFakeMessage(content)
        self.finish_reason = "stop"


class _CGFakeResp:
    def __init__(self, content, total, completion):
        self.choices = [_CGFakeChoice(content)]
        self.usage = _CGFakeUsage(total, completion)


class _CGFakeCompletions:
    def __init__(self, plan):
        self.plan = plan  # list of (content, total_tokens, completion_tokens)
        self.n = 0

    def create(self, **kwargs):
        content, total, completion = self.plan[self.n]
        self.n += 1
        return _CGFakeResp(content, total, completion)


class _CGFakeChat:
    def __init__(self, plan):
        self.completions = _CGFakeCompletions(plan)


class _CGFakeClient:
    def __init__(self, plan):
        self.chat = _CGFakeChat(plan)


# Call 1: total_tokens=900 (would fit fine under a real TPM budget) but completion_tokens=800
# specifically - proving otpm_pacer.record uses completion_tokens, NOT total_tokens.
# Call 2: requests another ~200 estimated output tokens - 800+200=1000 > 850 budget, so
# call_groq_with_retry must wait (via otpm_pacer's fake sleep) before making this call.
cg_client = _CGFakeClient([
    ('{"results": []}', 900, 800),
    ('{"results": []}', 300, 150),
])

call_groq_with_retry(cg_client, [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
                      max_completion_tokens=900, label="test-call-1", pacer=cg_tpm_pacer, estimated_tokens=900,
                      response_format={"type": "json_object"}, otpm_pacer=cg_otpm_pacer, estimated_output_tokens=800)

check("call_groq_with_retry: OTPM pacer recorded completion_tokens (800), not total_tokens (900)",
      cg_otpm_pacer.used_last_minute() == 800)
check("call_groq_with_retry: TPM pacer recorded total_tokens (900), independently of OTPM",
      cg_tpm_pacer.used_last_minute() == 900)
check("call_groq_with_retry: first call needed no wait (fresh OTPM budget)", len(shared_sleep_calls) == 0)

call_groq_with_retry(cg_client, [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
                      max_completion_tokens=300, label="test-call-2", pacer=cg_tpm_pacer, estimated_tokens=300,
                      response_format={"type": "json_object"}, otpm_pacer=cg_otpm_pacer, estimated_output_tokens=200)

check("call_groq_with_retry: second call correctly WAITED on the OTPM pacer (800+200 > 850 budget)",
      len(shared_sleep_calls) > 0)
check("call_groq_with_retry: after waiting, OTPM budget cleared enough for the second call's 200",
      cg_otpm_pacer.used_last_minute() + 200 <= 850 + 150)  # +150 already recorded by call 2 itself by now
check("call_groq_with_retry: second call's completion_tokens (150) recorded into OTPM pacer",
      150 in [t for _, t in cg_otpm_pacer.window])

# --------------------------------------------------------------------------
# 15. v2.5 hard per-request max_tokens cap - confirms no request process_group
#     actually builds can ever exceed MAX_TOKENS_PER_REQUEST_HARD_CAP (900),
#     the fix for a REAL rejected request that asked for 1,040.
# --------------------------------------------------------------------------
print("\n=== v2.5 hard per-request max_tokens cap ===")

check("v2.5 sanity: MAX_TOKENS_PER_REQUEST_HARD_CAP is 900 (100-token margin under Groq's ~1,000 hard ceiling)",
      MAX_TOKENS_PER_REQUEST_HARD_CAP == 900)

# 15a. Under DEFAULT settings specifically (batch_size=3, relations-only ceiling=300):
# the natural per-post ceiling * batch size should land EXACTLY on the hard cap, with no
# extra squeeze needed - the specific claim made in the v2.5 docstring section.
default_relations_only_max_tokens = min(
    ABSOLUTE_MAX_COMPLETION_TOKENS,
    PER_POST_COMPLETION_CEILING_RELATIONS * BATCH_SIZE_DEFAULT,
    MAX_TOKENS_PER_REQUEST_HARD_CAP,
)
check("v2.5: default batch_size(3) * relations-only ceiling(300) = 900, exactly the hard cap - no squeeze",
      default_relations_only_max_tokens == 900 and
      PER_POST_COMPLETION_CEILING_RELATIONS * BATCH_SIZE_DEFAULT == 900)


class _CapTrackingCompletions:
    """Records every max_completion_tokens this fake client is asked for."""
    def __init__(self):
        self.requested = []

    def create(self, **kwargs):
        self.requested.append(kwargs["max_completion_tokens"])
        # Return a trivially-valid empty-results response for every post_id asked about.
        import re
        user_msg = kwargs["messages"][1]["content"]
        post_ids = re.findall(r"post_id=(\S+?),", user_msg)
        results = [{"post_id": pid, "topic_relevant": True, "relations": []} for pid in post_ids]
        return _CGFakeResp(json.dumps({"results": results}), total=50, completion=10)


class _CapTrackingChat:
    def __init__(self):
        self.completions = _CapTrackingCompletions()


class _CapTrackingClient:
    def __init__(self):
        self.chat = _CapTrackingChat()


class _NoWaitPacer:
    """A pacer stub with an effectively infinite budget - isolates this test to just the
    max_tokens CAP, not pacing (already covered by sections 7/14)."""
    def wait_for_budget(self, *a, **kw):
        return False

    def record(self, *a, **kw):
        pass


# Stress a range of batch sizes (including ones larger than any default ever used) and
# BOTH modes, to prove the cap holds unconditionally - not just for today's defaults.
cap_client = _CapTrackingClient()
cap_pacer = _NoWaitPacer()
for stress_batch_size in (1, 2, 3, 4, 7, 10, 20):
    for stress_mode in ("full", "relations_only"):
        n_rows = stress_batch_size * 2  # a couple of batches' worth, so build_batches actually splits
        stress_df = pd.DataFrame({
            "id": [str(i) for i in range(n_rows)],
            "clean_text": ["short post text" for _ in range(n_rows)],
            "actors_mentioned": ["['Iran', 'Saudi Arabia']" for _ in range(n_rows)],
            "topic_relevant": pd.array([True] * n_rows, dtype="boolean"),
            "relations_checked": pd.array([pd.NA] * n_rows, dtype="boolean"),
        })

        class _StressArgs:
            batch_size = stress_batch_size
            max_batch_input_tokens = MAX_BATCH_INPUT_TOKENS_DEFAULT

        stress_counters = {"n_calls": 0, "n_rows_done": 0, "t_start": 0.0}
        rar.process_group(cap_client, stress_df, list(range(n_rows)), stress_mode, _StressArgs(),
                           cap_pacer, stress_counters)

over_cap = [r for r in cap_client.chat.completions.requested if r > MAX_TOKENS_PER_REQUEST_HARD_CAP]
check(f"v2.5: across {len(cap_client.chat.completions.requested)} constructed requests "
      f"(batch sizes 1-20, both modes), NONE exceeded the {MAX_TOKENS_PER_REQUEST_HARD_CAP} hard cap "
      f"(max seen: {max(cap_client.chat.completions.requested)})",
      len(over_cap) == 0)

# --------------------------------------------------------------------------
print(f"\n{'='*60}")
if failures:
    print(f"{len(failures)} check(s) FAILED:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("All checks passed. Batched prompt/parsing/pacing logic verified offline - safe to run against the real API.")

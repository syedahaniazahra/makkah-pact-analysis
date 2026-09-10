"""
extract_actor_mentions.py

Phase 6 (structural fix) - replaces the single-actor framing with a
proper multi-actor field. Scoped to the same detected_language=='en' rows
as the rest of Phase 5/6 for the per-row actors_mentioned field; the
aggregate outputs below are scoped narrower - see "Scoping" below.

The collection-time "matched_actor" column only records which actor's
SEARCH QUERY happened to surface a given tweet - not every actor the
tweet actually talks about. Since posts about the pact routinely name
multiple actors together (the three signatories especially), the
correlation analysis needs a field built from what's actually IN the
text, not from which query found it.

Adds to x_data_cleaned.csv (updated in place; "matched_actor" is left
completely untouched):
  actors_mentioned - list format, e.g. "['Pakistan', 'Saudi Arabia',
                      'Turkiye']" - every one of the 5 actors whose
                      keywords are found in clean_text, using the same
                      Unicode-normalized matching built for
                      actor_match_verified (handles Turkiye/Turkey/
                      Türkiye spellings). Computed for every English row
                      regardless of topic_relevant - this is a per-row
                      textual annotation ("does this post's text mention
                      Pakistan"), used elsewhere (e.g. the dashboard's
                      Timeline/Data Explorer) independent of relevance
                      filtering, so it is NOT narrowed the way the
                      aggregate outputs below are.

Separate outputs, English AND topic_relevant rows only (fixed 2026-09-10 -
see "Scoping"), RAW per-row counts - NOT cluster-deduplicated (same caveat
as hashtag_cooccurrence.csv; ask if you want a deduplicated version too):
  actor_frequency.csv    - actor, count
  actor_cooccurrence.csv - actor_1, actor_2, count - all 10 possible
                            pairs among the 5 actors, including 0-count
                            pairs, sorted by count descending.

Scoping (fixed 2026-09-10): actor_frequency.csv/actor_cooccurrence.csv
used to count every detected_language=='en' row, including the ~11-12%
that are topic_relevant==False (confirmed off-topic by
relevance_and_relations.py - the same class of contamination the
professor flagged in sentiment/hashtag numbers, see add_sentiment.py's
docstring for the fuller writeup). Now the two CSV aggregates only count
rows where topic_relevant==True, so an actor's mention count means "how
often discussed IN posts actually about the pact," not "how often
mentioned in any English post that happened to match a search query." Rows
that are topic_relevant==False or not yet classified this cycle (NaN) are
excluded from these two aggregate files, though their actors_mentioned
value is still written to x_data_cleaned.csv as normal (see above).
Same one-cycle-lag note as add_sentiment.py applies here too - this script
runs before relevance_and_relations.py in run_pipeline.py's step order, so
brand-new rows' topic_relevant is still NaN (and therefore excluded from
these two aggregates) until the NEXT pipeline cycle.

Note: these two files are not currently read by app.py (the dashboard's
Actor Network tab uses actor_network_v2.json/actor_centrality.csv from
build_network_v2.py, built from actor_relations.csv, which was already
topic_relevant-scoped before this fix). This fix is still worth making -
these are real output files someone could open directly - but it does not
by itself change anything currently visible in the dashboard.

Run:
    python extract_actor_mentions.py

Requires "clean_text" (written by preprocess_x_text.py) and
"topic_relevant" (clean_x_data.py / relevance_and_relations.py) to already
be present in x_data_cleaned.csv - see require_columns() below. Added
after a real scheduled-run incident where preprocess_x_text.py failed (a
missing NLTK data file) and this script then crashed with an unhandled
KeyError: 'clean_text' instead of skipping cleanly.
"""

import ast
import sys
import unicodedata
from collections import Counter
from itertools import combinations

import pandas as pd

# Preventive fix, applied pipeline-wide after a real incident: two other
# steps in this pipeline (preprocess_x_text.py, build_network_v2.py) each
# crashed with UnicodeEncodeError while printing scraped X/Twitter text
# (an emoji) to a Windows console defaulting to a single-byte codepage
# (cp1252) that can't represent it. This script processes the same scraped
# dataset on the same unattended schedule, so reconfiguring stdout/stderr
# to UTF-8 with errors="replace" here too closes off the same crash class
# pre-emptively - an unprintable character is swapped for a placeholder
# instead of ever being able to crash this step.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

INPUT_CSV = "x_data_cleaned.csv"
OUTPUT_CSV = "x_data_cleaned.csv"
ACTOR_FREQ_CSV = "actor_frequency.csv"
ACTOR_COOCCUR_CSV = "actor_cooccurrence.csv"

ACTORS = ["Iran", "Pakistan", "Saudi Arabia", "United States", "Turkiye"]

# Identical to extract_topics_hashtags.py's ACTOR_KEYWORDS, kept in sync
# on purpose so actors_mentioned and actor_match_verified/diff agree.
ACTOR_KEYWORDS = {
    "Iran": ["iran", "iranian"],
    "Pakistan": ["pakistan", "pakistani"],
    "Saudi Arabia": ["saudi arabia", "saudi"],
    "United States": ["united states", "u.s.", "usa"],
    "Turkiye": ["turkiye", "turkey", "turkish"],
}


def require_columns(df, columns, upstream_script):
    """Guards against the failure mode that motivated this check: a run
    where an upstream step (upstream_script) didn't complete and therefore
    never wrote a column this script depends on. Previously an unhandled
    KeyError here; now a clean SKIP_REASON + exit(3), which
    run_pipeline.py's run_step() recognizes as a graceful skip (same
    convention as preprocess_x_text.py's own NLTK self-heal skip) - the
    rest of the pipeline keeps going and this step retries next cycle."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        print(
            f"SKIP_REASON: column(s) {missing} not found in {INPUT_CSV} - "
            f"{upstream_script} has not completed successfully yet this cycle. "
            f"Nothing to do until it does; will retry next cycle."
        )
        sys.exit(3)


def strip_diacritics(s):
    """'türkiye' -> 'turkiye' - handles the official Turkish spelling
    (u+umlaut), common throughout this dataset."""
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def detect_actors_in_text(clean_text):
    text = strip_diacritics(clean_text or "")
    found = set()
    for actor, keywords in ACTOR_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            found.add(actor)
    return found


def main():
    df = pd.read_csv(INPUT_CSV, dtype={"id": str})
    require_columns(df, ["clean_text", "topic_relevant"], "preprocess_x_text.py / clean_x_data.py")
    en_mask = df["detected_language"] == "en"
    en_idx = df.index[en_mask]
    print(f"Working on {len(en_idx)} English rows (actors_mentioned column)")

    if "actors_mentioned" not in df.columns:
        df["actors_mentioned"] = ""

    df.loc[en_idx, "actors_mentioned"] = df.loc[en_idx, "clean_text"].apply(
        lambda t: str(sorted(detect_actors_in_text(t)))
    )
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved -> {OUTPUT_CSV} (matched_actor column untouched)")

    en_df = df.loc[en_idx].copy()
    en_df["_actor_set"] = en_df["actors_mentioned"].apply(ast.literal_eval)

    # Aggregates below are scoped to topic_relevant==True only - see this
    # module's "Scoping" docstring section for why.
    rel_df = en_df[en_df["topic_relevant"] == True]  # noqa: E712 - exact True, excludes False AND NaN
    print(
        f"Of those, {len(rel_df)} are topic_relevant=True - actor_frequency.csv/"
        f"actor_cooccurrence.csv are built from this narrower set, not all "
        f"{len(en_df)} English rows."
    )

    # --- actor_frequency.csv ---
    freq_counter = Counter()
    for s in rel_df["_actor_set"]:
        freq_counter.update(s)
    freq_df = pd.DataFrame(
        [{"actor": a, "count": freq_counter.get(a, 0)} for a in ACTORS]
    ).sort_values("count", ascending=False)
    freq_df.to_csv(ACTOR_FREQ_CSV, index=False)
    print(f"Saved -> {ACTOR_FREQ_CSV}")

    # --- actor_cooccurrence.csv (all 10 pairs, including 0-count) ---
    pair_counter = Counter()
    for s in rel_df["_actor_set"]:
        for a, b in combinations(sorted(s), 2):
            pair_counter[(a, b)] += 1

    all_pairs = list(combinations(sorted(ACTORS), 2))
    cooccur_df = pd.DataFrame([
        {"actor_1": a, "actor_2": b, "count": pair_counter.get((a, b), 0)}
        for a, b in all_pairs
    ]).sort_values("count", ascending=False)
    cooccur_df.to_csv(ACTOR_COOCCUR_CSV, index=False)
    print(f"Saved -> {ACTOR_COOCCUR_CSV}")

    print("\n=== Actor mention counts (English + topic_relevant subset, raw) ===")
    print(freq_df.to_string(index=False))
    print("\n=== Actor co-occurrence, all 10 pairs (English + topic_relevant subset, raw) ===")
    print(cooccur_df.to_string(index=False))


if __name__ == "__main__":
    main()

"""
extract_actor_mentions.py

Phase 6 (structural fix) - replaces the single-actor framing with a
proper multi-actor field. Scoped to the same 768 detected_language=='en'
rows as the rest of Phase 5/6.

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
                      Türkiye spellings).

Separate outputs (English subset only), RAW per-row counts - NOT
cluster-deduplicated (same caveat as hashtag_cooccurrence.csv; ask if
you want a deduplicated version too):
  actor_frequency.csv    - actor, count
  actor_cooccurrence.csv - actor_1, actor_2, count - all 10 possible
                            pairs among the 5 actors, including 0-count
                            pairs, sorted by count descending.

Run:
    python extract_actor_mentions.py
"""

import ast
import unicodedata
from collections import Counter
from itertools import combinations

import pandas as pd

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
    en_mask = df["detected_language"] == "en"
    en_idx = df.index[en_mask]
    print(f"Working on {len(en_idx)} English rows")

    if "actors_mentioned" not in df.columns:
        df["actors_mentioned"] = ""

    df.loc[en_idx, "actors_mentioned"] = df.loc[en_idx, "clean_text"].apply(
        lambda t: str(sorted(detect_actors_in_text(t)))
    )
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved -> {OUTPUT_CSV} (matched_actor column untouched)")

    en_df = df.loc[en_idx].copy()
    en_df["_actor_set"] = en_df["actors_mentioned"].apply(ast.literal_eval)

    # --- actor_frequency.csv ---
    freq_counter = Counter()
    for s in en_df["_actor_set"]:
        freq_counter.update(s)
    freq_df = pd.DataFrame(
        [{"actor": a, "count": freq_counter.get(a, 0)} for a in ACTORS]
    ).sort_values("count", ascending=False)
    freq_df.to_csv(ACTOR_FREQ_CSV, index=False)
    print(f"Saved -> {ACTOR_FREQ_CSV}")

    # --- actor_cooccurrence.csv (all 10 pairs, including 0-count) ---
    pair_counter = Counter()
    for s in en_df["_actor_set"]:
        for a, b in combinations(sorted(s), 2):
            pair_counter[(a, b)] += 1

    all_pairs = list(combinations(sorted(ACTORS), 2))
    cooccur_df = pd.DataFrame([
        {"actor_1": a, "actor_2": b, "count": pair_counter.get((a, b), 0)}
        for a, b in all_pairs
    ]).sort_values("count", ascending=False)
    cooccur_df.to_csv(ACTOR_COOCCUR_CSV, index=False)
    print(f"Saved -> {ACTOR_COOCCUR_CSV}")

    print("\n=== Actor mention counts (English subset, raw) ===")
    print(freq_df.to_string(index=False))
    print("\n=== Actor co-occurrence, all 10 pairs (raw) ===")
    print(cooccur_df.to_string(index=False))


if __name__ == "__main__":
    main()

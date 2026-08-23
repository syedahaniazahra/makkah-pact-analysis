"""
dedupe_actor_cooccurrence.py

Deduplicated versions of actor_frequency.csv / actor_cooccurrence.csv,
using the same "count each duplicate_content_cluster once" logic already
used for hashtag_frequency.csv's deduplicated_count. English subset
(768 rows) only, same as the rest of Phase 6.

Unit of counting: duplicate_content_cluster_id if the row is in a
cluster, else the row's own id (so non-clustered rows still count
individually). For each unit, we take the UNION of actors_mentioned
across every row in that unit (near-duplicate reposts should mention
the same actors; union is a safety margin for minor text variation
within a cluster) and count that unit once per actor / actor-pair it
contains - so a 7-tweet bot cluster contributes 1, not 7.

Outputs (alongside the original, non-deduplicated files, for comparison):
  actor_frequency_deduped.csv
  actor_cooccurrence_deduped.csv (all 10 pairs, including 0-count)

Run:
    python dedupe_actor_cooccurrence.py
"""

import ast
from collections import Counter
from itertools import combinations

import pandas as pd

INPUT_CSV = "x_data_cleaned.csv"
FREQ_DEDUP_CSV = "actor_frequency_deduped.csv"
COOCCUR_DEDUP_CSV = "actor_cooccurrence_deduped.csv"

ACTORS = ["Iran", "Pakistan", "Saudi Arabia", "United States", "Turkiye"]


def main():
    df = pd.read_csv(INPUT_CSV, dtype={"id": str})
    en_df = df[df["detected_language"] == "en"].copy()
    print(f"Working on {len(en_df)} English rows")

    en_df["_actor_set"] = en_df["actors_mentioned"].apply(
        lambda s: set(ast.literal_eval(s)) if isinstance(s, str) and s else set()
    )
    en_df["_count_unit"] = en_df.apply(
        lambda r: r["duplicate_content_cluster_id"]
        if pd.notna(r["duplicate_content_cluster_id"]) and str(r["duplicate_content_cluster_id"]).strip()
        else f"row_{r['id']}",
        axis=1,
    )

    # Union actors_mentioned across all rows sharing a count_unit.
    unit_actor_sets = {}
    for _, r in en_df.iterrows():
        unit_actor_sets.setdefault(r["_count_unit"], set()).update(r["_actor_set"])

    n_units = len(unit_actor_sets)
    n_clusters = en_df.loc[en_df["duplicate_content_cluster_id"].fillna("") != "", "duplicate_content_cluster_id"].nunique()
    print(f"{len(en_df)} rows collapse to {n_units} count units ({n_clusters} of which are duplicate-content clusters)")

    freq_counter = Counter()
    pair_counter = Counter()
    for actor_set in unit_actor_sets.values():
        freq_counter.update(actor_set)
        for a, b in combinations(sorted(actor_set), 2):
            pair_counter[(a, b)] += 1

    freq_df = pd.DataFrame(
        [{"actor": a, "deduplicated_count": freq_counter.get(a, 0)} for a in ACTORS]
    ).sort_values("deduplicated_count", ascending=False)
    freq_df.to_csv(FREQ_DEDUP_CSV, index=False)
    print(f"Saved -> {FREQ_DEDUP_CSV}")

    all_pairs = list(combinations(sorted(ACTORS), 2))
    cooccur_df = pd.DataFrame([
        {"actor_1": a, "actor_2": b, "deduplicated_count": pair_counter.get((a, b), 0)}
        for a, b in all_pairs
    ]).sort_values("deduplicated_count", ascending=False)
    cooccur_df.to_csv(COOCCUR_DEDUP_CSV, index=False)
    print(f"Saved -> {COOCCUR_DEDUP_CSV}")

    print("\n=== Deduplicated actor mention counts ===")
    print(freq_df.to_string(index=False))
    print("\n=== Deduplicated actor co-occurrence, all 10 pairs ===")
    print(cooccur_df.to_string(index=False))


if __name__ == "__main__":
    main()

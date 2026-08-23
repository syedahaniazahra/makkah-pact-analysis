"""
build_actor_network.py

Phase 7, step 1 - weighted actor network dataset for the Streamlit dashboard's
network graph. Pure reformatting of already-computed, already-deduplicated
numbers - no new analysis here.

Nodes: the 5 tracked actors, size = deduplicated_count from
       actor_frequency_deduped.csv (cluster-deduplicated mention count).
Edges: the 10 actor pairs, weight = deduplicated_count from
       actor_cooccurrence_deduped.csv (cluster-deduplicated co-occurrence
       count). All 10 pairs are included, even 0-weight ones, so the
       dashboard can render a complete graph.

Output: actor_network.json
  {
    "nodes": [{"id": "...", "label": "...", "size": N}, ...],
    "edges": [{"source": "...", "target": "...", "weight": N}, ...]
  }

Run:
    python build_actor_network.py
"""

import json

import pandas as pd

FREQ_CSV = "actor_frequency_deduped.csv"
COOCCUR_CSV = "actor_cooccurrence_deduped.csv"
OUTPUT_JSON = "actor_network.json"


def main():
    freq_df = pd.read_csv(FREQ_CSV)
    cooccur_df = pd.read_csv(COOCCUR_CSV)

    nodes = [
        {"id": row["actor"], "label": row["actor"], "size": int(row["deduplicated_count"])}
        for _, row in freq_df.iterrows()
    ]
    edges = [
        {
            "source": row["actor_1"],
            "target": row["actor_2"],
            "weight": int(row["deduplicated_count"]),
        }
        for _, row in cooccur_df.iterrows()
    ]

    network = {"nodes": nodes, "edges": edges}
    with open(OUTPUT_JSON, "w") as f:
        json.dump(network, f, indent=2)

    print(f"Saved -> {OUTPUT_JSON}")
    print(f"{len(nodes)} nodes, {len(edges)} edges")


if __name__ == "__main__":
    main()

"""
build_network_v2.py

Phase 6.5 -> network rebuild: turns actor_relations.csv (the typed,
per-pair relation_type extractions from the Groq pipeline) into a
weighted, typed actor network (actor_network_v2.json) plus real graph
centrality metrics (actor_centrality.csv), using networkx.

This replaces the earlier raw co-occurrence network (actor_cooccurrence.csv
/ actor_frequency.csv) with one grounded in the LLM-extracted relationship
TYPE between each pair, not just how often they were mentioned together.

Inputs (read-only, not modified):
    actor_relations.csv    - post_id, actor_1, actor_2, relation_type
    x_data_cleaned.csv     - used only to pull each mediating post's full
                             actors_mentioned list and clean_text snippet,
                             for the "which posts contain a mediating
                             relation" finding - not used for network math.

Outputs:
    actor_network_v2.json  - nodes (5 actors, sized by relation-row
                             involvement) + edges (10 pairs, each with a
                             full relation_type breakdown, total weight,
                             and dominant_relation_type)
    actor_centrality.csv   - actor, degree_centrality, betweenness_centrality

MEDIATING kept distinct (per instructions): it is never folded into
'unclear' in this dataset - every edge's relation_type breakdown lists it
as its own key even when its count is 0 for that pair, and every post
containing a mediating-typed relation is separately identified below.

Graph construction:
    Undirected graph, one node per actor (5 total). One edge per actor
    pair with weight = TOTAL relation-row count for that pair (all
    relation_types combined) - this is a complete graph (K5) since the
    real data has at least one relation for all 10 possible pairs.

Centrality choice, and why (read this before trusting the numbers):
    The task asks for centrality "based on the weighted edges." In a
    complete graph (every actor already connected to every other actor),
    UNWEIGHTED degree centrality (networkx's plain degree_centrality) is
    trivially 1.0 for all 5 actors - it can't distinguish anyone, because
    everyone is already connected to everyone. So "degree centrality" here
    is computed as WEIGHTED degree centrality (normalized node strength):
    each actor's total incident edge weight (sum of relation counts across
    all its pairs), divided by the total edge weight in the whole graph.
    This is NOT a single built-in networkx function (networkx has no
    canonical "weighted degree centrality") - it's the standard "strength"
    concept (Barrat et al.), computed here via G.degree(weight="weight")
    and normalized to sum to 1 across all 5 actors, so it reads like a
    share of total network activity.

    Betweenness centrality DOES have real weighted support in networkx,
    but networkx's weight parameter is interpreted as DISTANCE (higher
    weight = farther apart, on the logic that betweenness computes
    shortest paths). Our weight means the OPPOSITE - more relations
    between a pair means they're more strongly/closely connected, not
    farther apart. So a `distance` edge attribute of 1/weight is added
    (more relations -> smaller distance -> "closer"), and
    nx.betweenness_centrality(G, weight="distance") is used. This is the
    standard inversion trick for turning a "strength" weight into a
    "distance" weight for shortest-path-based metrics.
"""

import json
import os
import sys
from collections import Counter

import networkx as nx
import pandas as pd

# Real scheduled-run incident (the same bug class already fixed in
# preprocess_x_text.py, hit here too): this script's console report prints
# each mediating post's clean_text snippet (see "text: {m['text_snippet']!r}"
# in main() below) - real scraped X/Twitter text, which is routinely full of
# emoji. Windows' console defaults to a single-byte codepage (cp1252 here)
# that can't represent most emoji, so printing one crashes the whole
# process with UnicodeEncodeError - even though the actual JSON/CSV outputs
# above are written to disk successfully before this report ever prints, so
# the real network/centrality data was never at risk, just this diagnostic
# echo. Reconfiguring stdout/stderr to UTF-8 with errors="replace" makes
# every print() here safe regardless of console codepage - an unprintable
# character is swapped for a placeholder instead of raising. See
# preprocess_x_text.py's own copy of this same fix for the fuller
# explanation (first found there, with a different emoji, same root cause).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

RELATIONS_CSV = "actor_relations.csv"
X_DATA_CSV = "x_data_cleaned.csv"
NETWORK_JSON_OUT = "actor_network_v2.json"
CENTRALITY_CSV_OUT = "actor_centrality.csv"

# Fixed order for determinism in output (not alphabetical - matches the order
# the pact itself is usually discussed in: the three signatories, then the
# two most-referenced outside actors).
ALL_RELATION_TYPES = ["supportive", "hostile", "skeptical", "neutral-reporting", "mediating", "unclear"]


def pair_key(a1, a2):
    """Canonical, order-independent key for an actor pair."""
    return tuple(sorted([a1, a2]))


def dominant_type(breakdown):
    """Picks the relation_type with the highest count for an edge. Ties are
    broken by ALL_RELATION_TYPES order (stable, deterministic, and puts the
    more "decisive" categories - supportive/hostile/skeptical - ahead of
    neutral-reporting/mediating/unclear on an exact tie, so a genuine split
    reads as the more informative label rather than an arbitrary one)."""
    max_count = max(breakdown.values())
    for rtype in ALL_RELATION_TYPES:
        if breakdown.get(rtype, 0) == max_count:
            return rtype
    return None  # unreachable given breakdown is always built from ALL_RELATION_TYPES


def main():
    rel = pd.read_csv(RELATIONS_CSV, dtype={"post_id": str})
    df = pd.read_csv(X_DATA_CSV, dtype={"id": str})

    actors = sorted(set(rel["actor_1"]) | set(rel["actor_2"]))
    assert len(actors) == 5, f"expected 5 actors, found {len(actors)}: {actors}"

    # ---- Nodes: relation-row involvement (actor_1 OR actor_2 role) ----
    involvement = Counter()
    for col in ("actor_1", "actor_2"):
        involvement.update(rel[col].value_counts().to_dict())
    nodes = [
        {"id": actor, "label": actor, "relation_row_involvement": int(involvement[actor])}
        for actor in actors
    ]
    nodes.sort(key=lambda n: -n["relation_row_involvement"])

    # ---- Edges: per-pair relation_type breakdown + weight + dominant type ----
    pair_breakdown = {}
    for _, row in rel.iterrows():
        key = pair_key(row["actor_1"], row["actor_2"])
        pair_breakdown.setdefault(key, Counter())[row["relation_type"]] += 1

    edges = []
    G = nx.Graph()
    G.add_nodes_from(actors)
    for (a1, a2), counts in sorted(pair_breakdown.items()):
        breakdown = {rtype: int(counts.get(rtype, 0)) for rtype in ALL_RELATION_TYPES}
        weight = sum(breakdown.values())
        edges.append({
            "source": a1,
            "target": a2,
            "weight": weight,
            "relation_type_breakdown": breakdown,
            "dominant_relation_type": dominant_type(breakdown),
        })
        G.add_edge(a1, a2, weight=weight, distance=1.0 / weight)

    edges.sort(key=lambda e: -e["weight"])

    # ---- Mediating relations: kept distinct, never folded into unclear ----
    # actors_mentioned/clean_text are written by later pipeline steps
    # (extract_actor_mentions.py / preprocess_x_text.py) and aren't
    # required for the network/centrality math above - only for this
    # supplementary per-post enrichment. If either step failed to run
    # this cycle (an upstream failure, not this script's problem), degrade
    # this enrichment to None instead of crashing the whole network
    # rebuild with a KeyError over a field the core output doesn't even
    # need. Checked ONCE here rather than inside the loop.
    df_idx = df.set_index("id")
    has_actors_mentioned = "actors_mentioned" in df_idx.columns
    has_clean_text = "clean_text" in df_idx.columns
    if not has_actors_mentioned or not has_clean_text:
        missing = [c for c, present in
                   (("actors_mentioned", has_actors_mentioned), ("clean_text", has_clean_text))
                   if not present]
        print(f"Note: {missing} not found in {X_DATA_CSV} this cycle - "
              f"mediating_relations entries below will have those fields blank "
              f"(the network/centrality output above is unaffected).")

    mediating_rows = rel[rel["relation_type"] == "mediating"].copy()
    mediating_posts = []
    for _, r in mediating_rows.iterrows():
        pid = r["post_id"]
        actors_mentioned = None
        text_snippet = None
        if pid in df_idx.index:
            if has_actors_mentioned:
                raw_actors = df_idx.loc[pid, "actors_mentioned"]
                actors_mentioned = raw_actors if isinstance(raw_actors, str) else None
            if has_clean_text:
                text = df_idx.loc[pid, "clean_text"]
                if isinstance(text, str):
                    text_snippet = text[:300]
        mediating_posts.append({
            "post_id": pid,
            "actor_pair": [r["actor_1"], r["actor_2"]],
            "actors_mentioned_in_post": actors_mentioned,
            "text_snippet": text_snippet,
        })

    network = {
        "meta": {
            "source": RELATIONS_CSV,
            "total_relation_rows": int(len(rel)),
            "unique_posts_represented": int(rel["post_id"].nunique()),
            "node_count": len(nodes),
            "edge_count": len(edges),
            "notes": (
                "Weighted, typed actor network built from Groq-extracted per-pair "
                "relation_type data (replaces the earlier raw co-occurrence network). "
                "'mediating' is kept as its own category throughout, never folded "
                "into 'unclear'. See mediating_relations below for the specific "
                "posts/pairs where it appears."
            ),
        },
        "nodes": nodes,
        "edges": edges,
        "mediating_relations": mediating_posts,
    }

    # Atomic write (.tmp + os.replace): this now runs unattended on a
    # schedule (run_pipeline.py), not just by hand, so a crash or a killed
    # process mid-write must never leave a half-written JSON that the
    # dashboard could load and render as if it were complete. Same pattern
    # already used elsewhere in this project (see
    # purge_relations_for_post_ids in relevance_and_relations.py).
    tmp_json = NETWORK_JSON_OUT + ".tmp"
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(network, f, indent=2, ensure_ascii=False)
    os.replace(tmp_json, NETWORK_JSON_OUT)

    # ---- Centrality (see module docstring for the weighted-degree-centrality
    # and distance-inversion rationale) ----
    total_strength = sum(dict(G.degree(weight="weight")).values())
    degree_centrality = {
        actor: G.degree(actor, weight="weight") / total_strength
        for actor in actors
    }
    betweenness_centrality = nx.betweenness_centrality(G, weight="distance", normalized=True)

    centrality_df = pd.DataFrame([
        {
            "actor": actor,
            "degree_centrality": round(degree_centrality[actor], 6),
            "betweenness_centrality": round(betweenness_centrality[actor], 6),
        }
        for actor in actors
    ]).sort_values("degree_centrality", ascending=False)
    tmp_csv = CENTRALITY_CSV_OUT + ".tmp"
    centrality_df.to_csv(tmp_csv, index=False)
    os.replace(tmp_csv, CENTRALITY_CSV_OUT)

    # ---- Console report (for this script's own verification, and to hand
    # back exact numbers rather than re-deriving them by hand) ----
    print(f"Nodes ({len(nodes)}):")
    for n in nodes:
        print(f"  {n['id']:15s} involvement={n['relation_row_involvement']}")

    print(f"\nEdges ({len(edges)}), sorted by weight desc:")
    for e in edges:
        bd = e["relation_type_breakdown"]
        bd_str = ", ".join(f"{k}={v}" for k, v in bd.items() if v > 0)
        print(f"  {e['source']:15s} <-> {e['target']:15s} weight={e['weight']:4d}  dominant={e['dominant_relation_type']:18s}  [{bd_str}]")

    print(f"\nCentrality:")
    print(centrality_df.to_string(index=False))

    print(f"\nMediating relations ({len(mediating_posts)} post(s)):")
    for m in mediating_posts:
        print(f"  post_id={m['post_id']}  pair={m['actor_pair']}  actors_in_post={m['actors_mentioned_in_post']}")
        if m["text_snippet"]:
            print(f"    text: {m['text_snippet']!r}")

    print(f"\nWrote {NETWORK_JSON_OUT} and {CENTRALITY_CSV_OUT}")


if __name__ == "__main__":
    main()

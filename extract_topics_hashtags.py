"""
extract_topics_hashtags.py

Phase 6 - hashtag/topic extraction + actor-tag verification, scoped to
the 768 detected_language == 'en' rows in x_data_cleaned.csv (same scope
as Phase 5; non-English rows are left untouched).

Adds to x_data_cleaned.csv (updated in place):
  hashtags_extracted   - per row, whichever of {regex-on-text, existing
                          "hashtags" column} found MORE hashtags (list
                          format, e.g. "['#MakkahPact', '#Iran']").
  actor_match_verified - True/False: does an independent keyword check on
                          clean_text agree with the "matched_actor" column
                          set by the collection script?
  actor_match_diff     - (bonus) which actors differ between the two, for
                          quick debugging of any False rows.

Separate outputs:
  hashtag_frequency.csv    - hashtag, raw_count, deduplicated_count
                              (deduplicated_count treats every row in the
                              same duplicate_content_cluster as ONE vote,
                              so bot/aggregator clusters don't inflate it)
  hashtag_cooccurrence.csv - hashtag_1, hashtag_2, count (raw, per-post -
                              NOT cluster-deduplicated; flag if you want
                              that version too)

Run:
    python extract_topics_hashtags.py
"""

import ast
import re
import unicodedata
from collections import Counter
from itertools import combinations

import pandas as pd

INPUT_CSV = "x_data_cleaned.csv"
OUTPUT_CSV = "x_data_cleaned.csv"
FREQ_CSV = "hashtag_frequency.csv"
COOCCUR_CSV = "hashtag_cooccurrence.csv"

HASHTAG_RE = re.compile(r"#\w+")

# Same actor keyword definitions used at collection time (fetch_x_data.py)
# so this is an apples-to-apples cross-check of the same vocabulary,
# applied directly to the tweet text instead of trusting the search hit.
# Deliberately excludes bare "us" for United States - it's indistinguishable
# from the pronoun "us" and was never part of the original search terms
# either, so including it here would just manufacture false mismatches.
ACTOR_KEYWORDS = {
    "Iran": ["iran", "iranian"],
    "Pakistan": ["pakistan", "pakistani"],
    "Saudi Arabia": ["saudi arabia", "saudi"],
    "United States": ["united states", "u.s.", "usa"],
    "Turkiye": ["turkiye", "turkey", "turkish"],
}


def parse_existing_hashtags(hashtags_field):
    if not isinstance(hashtags_field, str) or not hashtags_field.strip():
        return set()
    return {h.strip() for h in hashtags_field.split(",") if h.strip()}


def regex_hashtags_from_text(text):
    return set(HASHTAG_RE.findall(text or ""))


def pick_more_complete(regex_set, existing_set):
    """Per-row: use whichever source found more hashtags; union on a tie
    so we don't arbitrarily throw away a hashtag only one source caught."""
    if len(regex_set) > len(existing_set):
        return regex_set
    if len(existing_set) > len(regex_set):
        return existing_set
    return regex_set | existing_set


def strip_diacritics(s):
    """'türkiye' -> 'turkiye'. Needed because a LOT of posts use the
    official Turkish spelling with u+umlaut, which plain ASCII substring
    matching would silently miss."""
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

    for col in ["hashtags_extracted", "actor_match_diff"]:
        if col not in df.columns:
            df[col] = ""
    if "actor_match_verified" not in df.columns:
        # object dtype (not pandas' strict "string" dtype) so it can hold
        # actual True/False booleans for English rows and stay blank/NA
        # for non-English rows without a dtype conflict.
        df["actor_match_verified"] = pd.array([pd.NA] * len(df), dtype="object")

    # --- Step 1: hashtags_extracted ---
    def build_hashtags_extracted(row):
        regex_set = regex_hashtags_from_text(row["text"])
        existing_set = parse_existing_hashtags(row.get("hashtags"))
        final = pick_more_complete(regex_set, existing_set)
        return str(sorted(final))

    df.loc[en_idx, "hashtags_extracted"] = df.loc[en_idx].apply(build_hashtags_extracted, axis=1)

    # --- Step 4: actor_match_verified ---
    def verify_actor(row):
        detected = detect_actors_in_text(row.get("clean_text", ""))
        matched = {a.strip() for a in str(row.get("matched_actor", "")).split(",") if a.strip()}
        verified = detected == matched
        diff = ""
        if not verified:
            only_matched = matched - detected
            only_detected = detected - matched
            parts = []
            if only_matched:
                parts.append(f"matched_actor has but text doesn't clearly show: {sorted(only_matched)}")
            if only_detected:
                parts.append(f"text shows but matched_actor missing: {sorted(only_detected)}")
            diff = "; ".join(parts)
        return pd.Series({"actor_match_verified": verified, "actor_match_diff": diff})

    verification = df.loc[en_idx].apply(verify_actor, axis=1)
    df.loc[en_idx, "actor_match_verified"] = verification["actor_match_verified"]
    df.loc[en_idx, "actor_match_diff"] = verification["actor_match_diff"]

    df.to_csv(OUTPUT_CSV, index=False)

    mismatches = (df.loc[en_idx, "actor_match_verified"] == False).sum()
    print(f"actor_match_verified == False: {mismatches} rows")
    print(f"Saved -> {OUTPUT_CSV}")

    # --- Step 2: hashtag_frequency.csv ---
    en_df = df.loc[en_idx].copy()
    en_df["_hashtag_set"] = en_df["hashtags_extracted"].apply(lambda s: set(ast.literal_eval(s)) if s else set())

    # count_unit: cluster id if clustered, else the row's own id (so every
    # non-clustered row still counts as its own independent unit)
    en_df["_count_unit"] = en_df.apply(
        lambda r: r["duplicate_content_cluster_id"]
        if pd.notna(r["duplicate_content_cluster_id"]) and str(r["duplicate_content_cluster_id"]).strip()
        else f"row_{r['id']}",
        axis=1,
    )

    raw_counter = Counter()
    dedup_units_per_hashtag = {}  # hashtag -> set of count_units
    for _, r in en_df.iterrows():
        for h in r["_hashtag_set"]:
            raw_counter[h] += 1
            dedup_units_per_hashtag.setdefault(h, set()).add(r["_count_unit"])

    freq_rows = [
        {"hashtag": h, "raw_count": raw_counter[h], "deduplicated_count": len(units)}
        for h, units in dedup_units_per_hashtag.items()
    ]
    freq_df = pd.DataFrame(freq_rows).sort_values("deduplicated_count", ascending=False)
    freq_df.to_csv(FREQ_CSV, index=False)
    print(f"Saved -> {FREQ_CSV} ({len(freq_df)} unique hashtags)")

    # --- Step 3: hashtag_cooccurrence.csv (raw, per-post, NOT cluster-deduped) ---
    pair_counter = Counter()
    for _, r in en_df.iterrows():
        tags = sorted(r["_hashtag_set"])
        for a, b in combinations(tags, 2):
            pair_counter[(a, b)] += 1

    cooccur_rows = [
        {"hashtag_1": a, "hashtag_2": b, "count": c}
        for (a, b), c in pair_counter.items()
    ]
    cooccur_df = pd.DataFrame(cooccur_rows).sort_values("count", ascending=False)
    cooccur_df.to_csv(COOCCUR_CSV, index=False)
    print(f"Saved -> {COOCCUR_CSV} ({len(cooccur_df)} unique pairs)")

    # --- Report ---
    print("\n=== Top 15 hashtags by deduplicated_count ===")
    print(freq_df.head(15).to_string(index=False))

    print("\n=== Top 10 hashtag co-occurrence pairs ===")
    print(cooccur_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

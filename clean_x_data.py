"""
clean_x_data.py

Phase 4 - data cleaning on x_data.csv. Adds flags/columns without deleting
or merging anything (except exact tweet-ID duplicates), so every decision
about what to exclude from later analysis stays reviewable. Reads
x_data.csv, writes x_data_cleaned.csv - the original file is never
modified.

What this adds:
  word_count               - word count of the tweet text
  long_form                - True if word_count > 300 (likely a dossier/
                              bulletin/transcript, not a typical post)
  duplicate_content_cluster_id
                            - groups near-duplicate text (>=90% similarity
                              after normalizing formatting) posted by
                              DIFFERENT accounts - e.g. copy-paste/viral
                              reposts. Blank if the row has no such match.
  detected_language         - langdetect's guess at the tweet's language
  flagged_content           - True if basic keyword screening flags
                              probable slurs/harassment (via the
                              better_profanity library's maintained word
                              list) or off-topic spam (crypto/sports/
                              generic religious-content promos). Not a
                              judgment call - just a first-pass filter for
                              human review before anything is excluded.
  flagged_reason             - which category tripped flagged_content
                              (profanity / crypto_spam / sports_spam /
                              religious_spam), for that review.
  confirmed_flag             - re-merged from flagged_review.csv (the
                              human review of flagged_content rows), by
                              id, if that file exists. See Step 5b below.

--- INCREMENTAL PROCESSING ---
Near-duplicate clustering and language detection are the two expensive
steps here - clustering in particular is O(n^2): comparing every row
against every other row. Reprocessing the ENTIRE historical dataset from
scratch on every refresh made refresh time grow with the total dataset
size, not with how many new posts were actually fetched - eventually
that's what caused a refresh to time out.

Fix: this script now diffs the incoming x_data.csv against whatever
x_data_cleaned.csv already has (by id) to split rows into "historical"
(already processed in a prior run) and "new" (never processed before).
  - detected_language: historical rows REUSE their previously-detected
    language untouched (their text hasn't changed, so there's no reason
    to redetect it). Only new rows run through langdetect.
  - duplicate_content_cluster_id: historical rows that already shared a
    cluster keep that grouping for free (no comparisons needed to
    rediscover it). The only fuzzy-match comparisons run are pairs where
    AT LEAST ONE side is a new row - this still catches a new post that's
    a near-duplicate of an OLD post (new-vs-historical), or of another
    new post (new-vs-new); it just skips historical-vs-historical pairs,
    since those were already decided and can't change (their text is
    fixed). This makes clustering cost scale with new-row-count x
    total-row-count instead of total-row-count^2 - a small refresh stays
    fast no matter how large the historical dataset has grown.
  - flagged_content / flagged_reason: also made incremental. Profiling
    turned up that this - NOT clustering - is actually the slowest step
    in the whole script (better_profanity's contains_profanity() check
    runs at roughly 90ms per row, ~800x slower per-row than the fuzzy
    clustering comparisons). It had the exact same "rescans everything
    every run" problem, so it gets the same fix: historical rows reuse
    their previous flagged_content/flagged_reason, only new rows are
    actually scanned.
  - word_count/long_form and the confirmed_flag merge are cheap, plain
    per-row logic with no pairwise comparisons and no slow library calls
    - re-running them over every row each time costs a few milliseconds
    total, so they're left as a full recompute for simplicity.

On the very first run (no x_data_cleaned.csv yet), every row is "new" -
this is identical to the old full-reprocessing behavior, just via the
same incremental code path with an empty history.

Run:
    python clean_x_data.py
"""

import os
import re

import pandas as pd
from rapidfuzz import fuzz
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from better_profanity import profanity

DetectorFactory.seed = 0  # make langdetect's output deterministic

INPUT_CSV = "x_data.csv"
OUTPUT_CSV = "x_data_cleaned.csv"
FLAGGED_REVIEW_CSV = "flagged_review.csv"

LONG_FORM_WORD_THRESHOLD = 300
SIMILARITY_THRESHOLD = 90  # rapidfuzz token_sort_ratio, 0-100
MIN_LEN_FOR_COMPARISON = 20  # skip trivially-short normalized text (e.g. just a URL/emoji)

# Basic keyword lists for the "off-topic spam" categories the user asked
# for. Deliberately loose/simple - flags for review, does not delete
# anything, and will have false positives (e.g. a tweet that mentions
# sports only in passing). better_profanity's own bundled word list
# handles slurs/harassment so we don't hand-author one here.
CRYPTO_SPAM_TERMS = [
    "airdrop", "presale", "$btc", "bitcoin giveaway", "crypto giveaway",
    "nft mint", "web3", "defi yield", "pump.fun", "100x gem", "to the moon",
]
SPORTS_SPAM_TERMS = [
    "touchdown", "premier league", "nba finals", "world cup goal",
    "fantasy football", "home run", "transfer window", "champions league",
]
RELIGIOUS_SPAM_TERMS = [
    "dua for", "quran recitation", "hajj packages", "umrah booking",
    "prayer times app", "daily islamic reminder", "subhanallah alhamdulillah",
]

URL_RE = re.compile(r"https?://\S+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
WHITESPACE_RE = re.compile(r"\s+")


def normalize_for_similarity(text):
    """Lowercase, strip URLs/punctuation, collapse whitespace - so minor
    formatting differences don't block a near-duplicate match."""
    text = (text or "").lower()
    text = URL_RE.sub("", text)
    text = NON_ALNUM_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def detect_language(text):
    text = (text or "").strip()
    if not text:
        return "unknown"
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def check_flagged(text):
    text_lower = (text or "").lower()
    if profanity.contains_profanity(text_lower):
        return True, "profanity"
    if any(term in text_lower for term in CRYPTO_SPAM_TERMS):
        return True, "crypto_spam"
    if any(term in text_lower for term in SPORTS_SPAM_TERMS):
        return True, "sports_spam"
    if any(term in text_lower for term in RELIGIOUS_SPAM_TERMS):
        return True, "religious_spam"
    return False, ""


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def load_previous_cleaned():
    """id -> {"detected_language", "duplicate_content_cluster_id"} from the
    x_data_cleaned.csv this script wrote LAST run, if any. Empty dict on
    the first-ever run (or if the previous file can't be read), which
    makes every row "new" - i.e. identical to full reprocessing."""
    if not os.path.exists(OUTPUT_CSV):
        return {}
    try:
        prev = pd.read_csv(
            OUTPUT_CSV, dtype={"id": str},
            usecols=lambda c: c in (
                "id", "detected_language", "duplicate_content_cluster_id",
                "flagged_content", "flagged_reason",
            ),
        )
    except Exception as exc:
        print(f"Could not read previous {OUTPUT_CSV} for incremental reuse ({exc}) - "
              f"falling back to full reprocessing of every row.")
        return {}
    lookup = {}
    for _, row in prev.iterrows():
        lookup[str(row["id"])] = {
            "detected_language": row.get("detected_language", ""),
            "duplicate_content_cluster_id": row.get("duplicate_content_cluster_id", ""),
            "flagged_content": row.get("flagged_content", False),
            "flagged_reason": row.get("flagged_reason", ""),
        }
    return lookup


def find_duplicate_clusters_incremental(df, is_new, prev_lookup):
    """Cluster rows whose normalized text is >=90% similar AND were posted
    by different accounts. Historical-vs-historical pairs are skipped
    entirely (already decided in a prior run); only pairs involving at
    least one new row are actually compared."""
    ids = df["id"].tolist()
    norm_texts = df["text"].apply(normalize_for_similarity).tolist()
    usernames = df["username"].tolist()
    n = len(df)
    uf = UnionFind(n)

    # Seed: pre-union historical rows that already shared a cluster id from
    # the previous run - reconstructs prior groupings for free, no fuzzy
    # comparisons needed for this part.
    prev_cluster_of = {}
    for i in range(n):
        if not is_new[i]:
            cid = prev_lookup.get(ids[i], {}).get("duplicate_content_cluster_id", "")
            if isinstance(cid, str) and cid.strip():
                prev_cluster_of[i] = cid.strip()
    by_prev_cluster = {}
    for i, cid in prev_cluster_of.items():
        by_prev_cluster.setdefault(cid, []).append(i)
    for members in by_prev_cluster.values():
        for j in members[1:]:
            uf.union(members[0], j)

    # Only compare pairs where at least one side is NEW. This still catches
    # a new post that's a near-duplicate of an old one (new-vs-historical)
    # or of another new one (new-vs-new); historical-vs-historical pairs
    # are skipped since they can't have changed.
    new_indices = [i for i in range(n) if is_new[i]]
    n_comparisons = 0
    for i in new_indices:
        if len(norm_texts[i]) < MIN_LEN_FOR_COMPARISON:
            continue
        for j in range(n):
            if j == i:
                continue
            if is_new[j] and j <= i:
                continue  # avoid double-counting a new-vs-new pair
            if len(norm_texts[j]) < MIN_LEN_FOR_COMPARISON:
                continue
            if usernames[i] == usernames[j]:
                continue
            n_comparisons += 1
            score = fuzz.token_sort_ratio(norm_texts[i], norm_texts[j])
            if score >= SIMILARITY_THRESHOLD:
                uf.union(i, j)

    # Build final cluster assignment - keep an existing cluster's id where
    # possible (a new row joining cluster_7 should still be called
    # cluster_7, not get renumbered). If a new row happens to bridge two
    # PREVIOUSLY SEPARATE historical clusters together, merge them under
    # the lower-numbered id and note it.
    root_to_members = {}
    for i in range(n):
        root_to_members.setdefault(uf.find(i), []).append(i)

    def cluster_num(cid):
        try:
            return int(cid.split("_")[1])
        except (IndexError, ValueError):
            return -1

    existing_nums = [cluster_num(cid) for cid in prev_cluster_of.values()]
    next_cluster_num = (max(existing_nums) + 1) if existing_nums else 1

    cluster_id_col = [""] * n
    cluster_sizes = {}
    merge_notes = []
    for root, members in root_to_members.items():
        if len(members) < 2:
            continue
        existing_ids_here = sorted(
            {prev_cluster_of[i] for i in members if i in prev_cluster_of}, key=cluster_num
        )
        if existing_ids_here:
            cid = existing_ids_here[0]
            if len(existing_ids_here) > 1:
                merge_notes.append(f"{', '.join(existing_ids_here)} merged into {cid} (a new post links them)")
        else:
            cid = f"cluster_{next_cluster_num}"
            next_cluster_num += 1
        for i in members:
            cluster_id_col[i] = cid
        cluster_sizes[cid] = len(members)

    return cluster_id_col, cluster_sizes, merge_notes, n_comparisons


def main():
    df = pd.read_csv(INPUT_CSV, dtype={"id": str})
    total_before = len(df)

    # --- Step 1: exact duplicate rows (same tweet id) ---
    exact_dupes = df["id"].duplicated().sum()
    df = df.drop_duplicates(subset="id", keep="first").reset_index(drop=True)

    # --- Step 1b: split into historical (already processed before) vs new ---
    prev_lookup = load_previous_cleaned()
    is_new = [rid not in prev_lookup for rid in df["id"]]
    n_new = sum(is_new)
    n_historical = len(df) - n_new
    if prev_lookup:
        print(f"Incremental run: {n_historical} historical row(s) reusing prior language/"
              f"cluster results, {n_new} new row(s) to actually process.")
    else:
        print(f"No previous {OUTPUT_CSV} found - treating all {n_new} row(s) as new "
              f"(first run, or previous output unreadable).")

    # --- Step 6: standardize timestamp to one consistent format ---
    parsed_ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    missing_timestamps = parsed_ts.isna().sum()
    df["timestamp"] = parsed_ts.dt.strftime("%Y-%m-%d %H:%M:%S%z")

    # --- Step 7: data-quality checks (reported, not necessarily fixed) ---
    text_filled = df["text"].fillna("")
    empty_text_rows = (text_filled.str.strip() == "").sum()
    # "Obviously broken/garbled": essentially no alphabetic content at all
    # (e.g. encoding artifacts, control characters, or emoji/symbol-only
    # noise) in an otherwise non-empty row.
    def looks_garbled(t):
        t = t or ""
        if t.strip() == "":
            return False
        letters = sum(c.isalpha() for c in t)
        return letters / max(len(t), 1) < 0.15
    garbled_rows = text_filled.apply(looks_garbled).sum()

    # --- Step 2: word_count + long_form flag (cheap, full recompute) ---
    df["word_count"] = text_filled.apply(lambda t: len(t.split()))
    df["long_form"] = df["word_count"] > LONG_FORM_WORD_THRESHOLD

    # --- Step 3: near-duplicate/copy-paste content clustering (INCREMENTAL) ---
    print("Checking for near-duplicate content across accounts "
          "(incremental - only new rows are compared)...")
    cluster_ids, cluster_sizes, merge_notes, n_comparisons = find_duplicate_clusters_incremental(
        df, is_new, prev_lookup
    )
    df["duplicate_content_cluster_id"] = cluster_ids
    print(f"  {n_comparisons} pairwise comparison(s) run (a full reprocess of all "
          f"{len(df)} rows would run up to ~{len(df) * (len(df) - 1) // 2}).")
    if merge_notes:
        print("  Cluster merges from new rows bridging old clusters:")
        for note in merge_notes:
            print(f"    {note}")

    # --- Step 4: detected_language (INCREMENTAL - only new rows) ---
    df["detected_language"] = ""
    hist_idx = [i for i in range(len(df)) if not is_new[i]]
    new_idx = [i for i in range(len(df)) if is_new[i]]
    if hist_idx:
        df.loc[hist_idx, "detected_language"] = [
            prev_lookup[df.loc[i, "id"]]["detected_language"] for i in hist_idx
        ]
    if new_idx:
        print(f"Detecting language for {len(new_idx)} new row(s) only "
              f"({len(hist_idx)} historical row(s) reused their prior result)...")
        df.loc[new_idx, "detected_language"] = text_filled.loc[new_idx].apply(detect_language)
    else:
        print("No new rows - skipping language detection entirely.")

    # --- Step 5: flagged_content (INCREMENTAL - only new rows) ---
    # Profiling this while building the incremental clustering fix found
    # that check_flagged (via better_profanity's contains_profanity) is
    # actually the SLOWEST step in this whole script by a wide margin -
    # roughly 90ms per row, dwarfing the clustering/language-detection cost
    # this incremental rework originally targeted. It has the exact same
    # "rescans the entire historical dataset every run" problem, so it gets
    # the same fix: historical rows reuse their previously-computed
    # flagged_content/flagged_reason, only new rows are actually scanned.
    df["flagged_content"] = False
    df["flagged_reason"] = ""
    if hist_idx:
        df.loc[hist_idx, "flagged_content"] = [
            prev_lookup[df.loc[i, "id"]]["flagged_content"] for i in hist_idx
        ]
        df.loc[hist_idx, "flagged_reason"] = [
            prev_lookup[df.loc[i, "id"]]["flagged_reason"] for i in hist_idx
        ]
    if new_idx:
        flags_new = text_filled.loc[new_idx].apply(check_flagged)
        df.loc[new_idx, "flagged_content"] = flags_new.apply(lambda x: x[0])
        df.loc[new_idx, "flagged_reason"] = flags_new.apply(lambda x: x[1])

    # --- Step 5b: re-merge confirmed_flag from the human review file ---
    # This file rebuilds from x_data.csv from scratch every run, so this
    # merge is what keeps a re-run from silently losing prior manual review.
    n_confirmed_real = 0
    if os.path.exists(FLAGGED_REVIEW_CSV):
        review = pd.read_csv(FLAGGED_REVIEW_CSV, dtype={"id": str})
        review_map = dict(zip(review["id"], review["confirmed_flag"]))
        df["confirmed_flag"] = df["id"].map(review_map)  # NaN for anything not reviewed
        n_confirmed_real = int((df["confirmed_flag"] == True).sum())  # noqa: E712
        print(f"Merged {FLAGGED_REVIEW_CSV}: {len(review)} reviewed row(s), "
              f"{n_confirmed_real} confirmed real -> excluded downstream.")
    else:
        df["confirmed_flag"] = pd.array([pd.NA] * len(df), dtype="object")
        print(f"{FLAGGED_REVIEW_CSV} not found - confirmed_flag left blank for all rows "
              f"(nothing has been through manual review yet).")

    # --- Step 8: save (original x_data.csv is never touched) ---
    df.to_csv(OUTPUT_CSV, index=False)

    # --- Step 9: report ---
    print("\n=== Phase 4 cleaning report ===")
    print(f"Rows before: {total_before}")
    print(f"Exact duplicate id rows removed: {exact_dupes}")
    print(f"Total rows in {OUTPUT_CSV}: {len(df)}")
    print(f"New rows processed this run: {n_new} (historical reused: {n_historical})")
    print()
    print(f"long_form (>{LONG_FORM_WORD_THRESHOLD} words): {int(df['long_form'].sum())}")
    print()
    rows_in_clusters = sum(1 for c in cluster_ids if c)
    print(f"Rows in a duplicate-content cluster: {rows_in_clusters}")
    print(f"Number of clusters: {len(cluster_sizes)}")
    if cluster_sizes:
        top = sorted(cluster_sizes.items(), key=lambda kv: kv[1], reverse=True)[:5]
        print(f"Largest clusters: {top}")
    print()
    lang_counts = df["detected_language"].value_counts().head(5)
    print("Top 5 detected languages:")
    for lang, count in lang_counts.items():
        print(f"  {lang}: {count}")
    print()
    print(f"flagged_content rows: {int(df['flagged_content'].sum())}")
    if df["flagged_content"].sum():
        print("  by reason:")
        for reason, count in df.loc[df["flagged_content"], "flagged_reason"].value_counts().items():
            print(f"    {reason}: {count}")
    print(f"confirmed_flag == True (manually confirmed, excluded downstream): {n_confirmed_real}")
    print()
    print(f"Empty text rows: {empty_text_rows}")
    print(f"Missing/unparseable timestamps: {missing_timestamps}")
    print(f"Obviously garbled rows (little/no alphabetic content): {garbled_rows}")


if __name__ == "__main__":
    main()

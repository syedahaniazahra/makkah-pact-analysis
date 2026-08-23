"""
preprocess_x_text.py

Phase 5 - text preprocessing, scoped to the 768 rows where
detected_language == "en" in x_data_cleaned.csv. Adds:

  clean_text               - lowercased, URLs/@mentions/RT-artifacts
                              removed, hashtag WORDS kept (only the '#'
                              symbol is stripped) since Phase 6 needs them.
  clean_text_lemmatized     - clean_text tokenized, English stopwords
                              removed, lemmatized (NLTK WordNet), space-
                              joined back into a string.
  long_form_excerpt         - for long_form=True rows only: first 100
                              words of clean_text (short version for
                              charts/tables that can't fit the full text).

Non-English rows are left completely untouched (all new columns blank
for them) - this script only ever narrows to language == "en" for
processing, never drops or reorders any row.

The original "text" column is never modified. Per the user: rows with
confirmed_flag == True (the 12 manually-confirmed slur/harassment hits
from the flagged-content review) should be excluded from any quoting or
display - this script doesn't touch that column, just carries it through.

Run:
    python preprocess_x_text.py
"""

import re

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

INPUT_CSV = "x_data_cleaned.csv"
OUTPUT_CSV = "x_data_cleaned.csv"  # updated in place, adding columns

LONG_FORM_EXCERPT_WORDS = 100

URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")
RT_RE = re.compile(r"\brt\b\s*:?", re.IGNORECASE)
HASHTAG_SYMBOL_RE = re.compile(r"#(\w+)")  # drop '#', KEEP the word
WHITESPACE_RE = re.compile(r"\s+")

STOPWORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def make_clean_text(text):
    t = (text or "").lower()
    t = URL_RE.sub("", t)
    t = MENTION_RE.sub("", t)
    t = RT_RE.sub("", t)
    t = HASHTAG_SYMBOL_RE.sub(r"\1", t)  # "#makkahpact" -> "makkahpact"
    t = WHITESPACE_RE.sub(" ", t).strip()
    return t


def lemmatize_clean_text(clean_text):
    tokens = word_tokenize(clean_text)
    kept = [w for w in tokens if w.isalpha() and w not in STOPWORDS]
    lemmas = []
    for w in kept:
        # WordNetLemmatizer's default noun-POS assumption mangles some
        # short words - notably "us" -> "u" (it treats it as a plural of
        # "u"), which matters here since "US" is one of the 5 tracked
        # actors. Skip lemmatization for very short tokens; there's
        # essentially nothing useful to lemmatize at that length anyway.
        lemmas.append(w if len(w) <= 2 else lemmatizer.lemmatize(w))
    return " ".join(lemmas)


def main():
    df = pd.read_csv(INPUT_CSV, dtype={"id": str})

    en_mask = df["detected_language"] == "en"
    n_en = en_mask.sum()
    print(f"English rows (detected_language == 'en'): {n_en}")

    # Initialize new columns as blank for ALL rows (non-English rows stay
    # untouched), then fill in only for the English subset.
    for col in ["clean_text", "clean_text_lemmatized", "long_form_excerpt"]:
        if col not in df.columns:
            df[col] = ""

    en_idx = df.index[en_mask]
    df.loc[en_idx, "clean_text"] = df.loc[en_idx, "text"].apply(make_clean_text)
    df.loc[en_idx, "clean_text_lemmatized"] = df.loc[en_idx, "clean_text"].apply(lemmatize_clean_text)

    long_form_en = en_idx[df.loc[en_idx, "long_form"] == True]
    df.loc[long_form_en, "long_form_excerpt"] = df.loc[long_form_en, "clean_text"].apply(
        lambda t: " ".join(t.split()[:LONG_FORM_EXCERPT_WORDS])
    )

    df.to_csv(OUTPUT_CSV, index=False)

    print(f"long_form rows within English subset: {len(long_form_en)}")
    print(f"Saved -> {OUTPUT_CSV}")

    # --- Step 6: one example per duplicate-content cluster, English subset only ---
    en_df = df.loc[en_idx]
    clustered_en = en_df[en_df["duplicate_content_cluster_id"].fillna("") != ""]
    all_clusters = df[df["duplicate_content_cluster_id"].fillna("") != ""]["duplicate_content_cluster_id"].nunique()
    clusters_in_en = clustered_en["duplicate_content_cluster_id"].nunique()
    print(f"\nDuplicate-content clusters overall: {all_clusters}; "
          f"clusters with >=1 row in the English subset: {clusters_in_en}")

    print("\n=== One example per cluster (English subset) ===")
    reviewable = clustered_en[clustered_en["confirmed_flag"] != True]  # honor the exclusion
    for cid, group in reviewable.groupby("duplicate_content_cluster_id"):
        row = group.iloc[0]
        size = (df["duplicate_content_cluster_id"] == cid).sum()
        snippet = str(row["text"]).replace("\n", " ")[:140]
        print(f"[{cid}] size={size} matched_actor={row['matched_actor']} "
              f"user=@{row['username']}: {snippet}...")


if __name__ == "__main__":
    main()

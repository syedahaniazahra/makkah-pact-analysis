"""
add_sentiment.py

Phase 7, step 4 - VADER sentiment on clean_text, English rows only.

Adds to x_data_cleaned.csv (updated in place; non-English rows left blank,
same scoping convention as every prior phase):
  sentiment_score - VADER compound score, -1 (most negative) to +1 (most
                     positive).
  sentiment_label - "positive" if score >= 0.05, "negative" if <= -0.05,
                     else "neutral" (VADER's own documented thresholds).

Then reports, per actor (using "actors_mentioned" - a post naming several
actors contributes to each of them):
  - average sentiment_score
  - average likes, average retweets
so the two can be compared side by side per actor.

Run:
    python add_sentiment.py
"""

import ast

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

INPUT_CSV = "x_data_cleaned.csv"
OUTPUT_CSV = "x_data_cleaned.csv"

ACTORS = ["Iran", "Pakistan", "Saudi Arabia", "United States", "Turkiye"]

analyzer = SentimentIntensityAnalyzer()


def label_from_score(score):
    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "negative"
    return "neutral"


def main():
    df = pd.read_csv(INPUT_CSV, dtype={"id": str})
    en_mask = df["detected_language"] == "en"
    en_idx = df.index[en_mask]
    print(f"Working on {len(en_idx)} English rows")

    for col in ["sentiment_score", "sentiment_label"]:
        if col not in df.columns:
            df[col] = ""
    df["sentiment_score"] = pd.array([pd.NA] * len(df), dtype="object")

    scores = df.loc[en_idx, "clean_text"].apply(
        lambda t: analyzer.polarity_scores(t or "")["compound"]
    )
    df.loc[en_idx, "sentiment_score"] = scores
    df.loc[en_idx, "sentiment_label"] = scores.apply(label_from_score)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved -> {OUTPUT_CSV}")

    print("\n=== sentiment_label distribution (English subset) ===")
    print(df.loc[en_idx, "sentiment_label"].value_counts().to_string())

    # --- per-actor sentiment vs engagement ---
    en_df = df.loc[en_idx].copy()
    en_df["_actor_set"] = en_df["actors_mentioned"].apply(
        lambda s: set(ast.literal_eval(s)) if isinstance(s, str) and s else set()
    )
    en_df["sentiment_score"] = pd.to_numeric(en_df["sentiment_score"], errors="coerce")
    en_df["likes"] = pd.to_numeric(en_df["likes"], errors="coerce")
    en_df["retweets"] = pd.to_numeric(en_df["retweets"], errors="coerce")

    rows = []
    for actor in ACTORS:
        mask = en_df["_actor_set"].apply(lambda s: actor in s)
        subset = en_df[mask]
        rows.append({
            "actor": actor,
            "n_posts": len(subset),
            "avg_sentiment": round(subset["sentiment_score"].mean(), 3) if len(subset) else None,
            "avg_likes": round(subset["likes"].mean(), 1) if len(subset) else None,
            "avg_retweets": round(subset["retweets"].mean(), 1) if len(subset) else None,
        })
    report_df = pd.DataFrame(rows).sort_values("avg_sentiment")
    print("\n=== Sentiment vs engagement, per actor (English subset) ===")
    print(report_df.to_string(index=False))


if __name__ == "__main__":
    main()

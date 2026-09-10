"""
add_sentiment.py

Phase 7, step 4 - VADER sentiment on clean_text, English AND
topic-relevant rows only.

Adds to x_data_cleaned.csv (updated in place; non-English rows AND
off-topic/not-yet-classified rows left blank - see "Scoping" below):
  sentiment_score - VADER compound score, -1 (most negative) to +1 (most
                     positive).
  sentiment_label - "positive" if score >= 0.05, "negative" if <= -0.05,
                     else "neutral" (VADER's own documented thresholds).

Then reports, per actor (using "actors_mentioned" - a post naming several
actors contributes to each of them):
  - average sentiment_score
  - average likes, average retweets
so the two can be compared side by side per actor.

Scoping (fixed 2026-09-10): this used to run on every detected_language=='en'
row, full stop. The professor flagged that sentiment/hashtag numbers looked
"not relevant or in line with the Mecca Pact" - traced to this: ~11-12% of
the English rows are topic_relevant==False (confirmed off-topic by
relevance_and_relations.py's Groq classification, e.g. incidental "Iran"/
"Turkey" mentions unrelated to the pact), and their sentiment was being
averaged in right alongside the genuinely on-topic posts, diluting every
per-actor average. Now scoped to detected_language=='en' AND
topic_relevant==True - matching the same row set actor_relations.csv and
the Network tab are already built from (relevance_and_relations.py), so
sentiment numbers are finally answering the same question as the rest of
the analysis: "what does on-topic discussion look like," not "what does
all English-language chatter that happened to mention a keyword look
like." Rows that are topic_relevant==False, OR not yet classified this
cycle (NaN - relevance_and_relations.py hasn't reached them yet), get
sentiment_score/sentiment_label left blank, same convention as non-English
rows.

One real consequence worth knowing about: in run_pipeline.py's step order,
this script (step 6) runs BEFORE relevance_and_relations.py (step 7) - so
brand-new rows fetched THIS cycle still have topic_relevant==NaN when this
script sees them, and get skipped (blank sentiment) for this cycle. They
pick up topic_relevant (True/False) from relevance_and_relations.py later
in the SAME cycle, and clean_x_data.py's "Step 5c" preserves that value
into the NEXT cycle's x_data_cleaned.csv rebuild - so a genuinely
topic_relevant row's sentiment is computed exactly one pipeline cycle
after it's fetched, not immediately. This is not a bug to "fix" by
reordering the pipeline (a bigger, riskier change than this scoping fix
warrants) - it's the same "still catching up, will resolve next cycle"
situation app.py's own data-completeness check already detects and
reports (see app.py's _incomplete/_sentiment_missing warning).

Run:
    python add_sentiment.py

Requires "clean_text" (preprocess_x_text.py), "actors_mentioned"
(extract_actor_mentions.py), and "topic_relevant" (clean_x_data.py /
relevance_and_relations.py) to already be present - see
require_columns() below. Added after a real scheduled-run incident where
an upstream step failed and this script then crashed with an unhandled
KeyError instead of skipping cleanly.
"""

import ast
import sys

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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

ACTORS = ["Iran", "Pakistan", "Saudi Arabia", "United States", "Turkiye"]

analyzer = SentimentIntensityAnalyzer()


def require_columns(df, columns, upstream_script):
    """See extract_actor_mentions.py's require_columns() for the full
    rationale - same convention, same SKIP_REASON + exit(3) contract that
    run_pipeline.py's run_step() recognizes as a graceful skip."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        print(
            f"SKIP_REASON: column(s) {missing} not found in {INPUT_CSV} - "
            f"{upstream_script} has not completed successfully yet this cycle. "
            f"Nothing to do until it does; will retry next cycle."
        )
        sys.exit(3)


def label_from_score(score):
    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "negative"
    return "neutral"


def main():
    df = pd.read_csv(INPUT_CSV, dtype={"id": str})
    require_columns(df, ["clean_text", "actors_mentioned", "topic_relevant"],
                     "preprocess_x_text.py / extract_actor_mentions.py / "
                     "clean_x_data.py")
    en_mask = df["detected_language"] == "en"
    relevant_mask = en_mask & (df["topic_relevant"] == True)  # noqa: E712 - exact True, excludes False AND NaN
    relevant_idx = df.index[relevant_mask]
    n_en = int(en_mask.sum())
    n_off_topic = int((en_mask & (df["topic_relevant"] == False)).sum())  # noqa: E712
    n_unclassified = int((en_mask & df["topic_relevant"].isna()).sum())
    print(
        f"{n_en} English rows total: {len(relevant_idx)} topic_relevant=True "
        f"(sentiment computed for these), {n_off_topic} topic_relevant=False "
        f"(left blank), {n_unclassified} not yet classified this cycle (left "
        f"blank, will pick up sentiment next cycle once relevance_and_relations.py "
        f"reaches them)."
    )

    for col in ["sentiment_score", "sentiment_label"]:
        if col not in df.columns:
            df[col] = ""
    df["sentiment_score"] = pd.array([pd.NA] * len(df), dtype="object")
    df["sentiment_label"] = pd.array([pd.NA] * len(df), dtype="object")

    scores = df.loc[relevant_idx, "clean_text"].apply(
        lambda t: analyzer.polarity_scores(t or "")["compound"]
    )
    df.loc[relevant_idx, "sentiment_score"] = scores
    df.loc[relevant_idx, "sentiment_label"] = scores.apply(label_from_score)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved -> {OUTPUT_CSV}")

    print("\n=== sentiment_label distribution (English + topic_relevant subset) ===")
    print(df.loc[relevant_idx, "sentiment_label"].value_counts().to_string())

    # --- per-actor sentiment vs engagement ---
    rel_df = df.loc[relevant_idx].copy()
    rel_df["_actor_set"] = rel_df["actors_mentioned"].apply(
        lambda s: set(ast.literal_eval(s)) if isinstance(s, str) and s else set()
    )
    rel_df["sentiment_score"] = pd.to_numeric(rel_df["sentiment_score"], errors="coerce")
    rel_df["likes"] = pd.to_numeric(rel_df["likes"], errors="coerce")
    rel_df["retweets"] = pd.to_numeric(rel_df["retweets"], errors="coerce")

    rows = []
    for actor in ACTORS:
        mask = rel_df["_actor_set"].apply(lambda s: actor in s)
        subset = rel_df[mask]
        rows.append({
            "actor": actor,
            "n_posts": len(subset),
            "avg_sentiment": round(subset["sentiment_score"].mean(), 3) if len(subset) else None,
            "avg_likes": round(subset["likes"].mean(), 1) if len(subset) else None,
            "avg_retweets": round(subset["retweets"].mean(), 1) if len(subset) else None,
        })
    report_df = pd.DataFrame(rows).sort_values("avg_sentiment")
    print("\n=== Sentiment vs engagement, per actor (English + topic_relevant subset) ===")
    print(report_df.to_string(index=False))


if __name__ == "__main__":
    main()

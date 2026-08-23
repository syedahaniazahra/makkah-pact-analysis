"""
fetch_x_data.py

Phase 3 - collection run. Searches X for the Makkah/Mecca pact keyword set
combined with each of the 5 actor countries (Iran, Pakistan, Saudi Arabia,
United States, Turkiye), and saves the combined, deduplicated results to
x_data.csv.

Two modes:
  Full (default) - since 2026-08-03 (a few days before the pact was signed
      on Aug 7, to capture the lead-up period too), capped at 300 tweets
      per actor. This is the historical backfill - normally a one-time run.
  --recent (used by the dashboard's Refresh Data button) - only looks back
      RECENT_DAYS days and caps at RECENT_LIMIT_PER_ACTOR tweets/actor, so
      a "what's new" refresh is a small, fast pull rather than re-scraping
      the whole window every time. Rate-limit delays (ACTOR_DELAY_*,
      INTRA_SEARCH_DELAY_*) are unchanged in either mode - recent mode is
      faster only because there's less to fetch, not because it waits less.

Requires: setup_x_account.py already run successfully (logged-in account
in accounts.db in this folder).

Prints a "NEW_ROWS_ADDED:<n>" line at the end - tooling (e.g. the
dashboard) that wants the count of genuinely new rows this run added can
parse that line rather than re-diffing the CSV itself.

Run:
    python fetch_x_data.py                # full historical pull
    python fetch_x_data.py --recent        # quick refresh, last 3 days, 40/actor
    python fetch_x_data.py --recent --recent-days 2 --recent-limit 30
"""

import argparse
import asyncio
import datetime as dt
import os
import random
import re
import time

import pandas as pd
from twscrape import API

OUTPUT_CSV = "x_data.csv"

# Pact was signed Aug 7, 2026 - start a few days earlier to also capture
# the lead-up/rumor period, not just the aftermath. Overridden at runtime
# in --recent mode (see parse_args/main).
SINCE_DATE = "2026-08-03"
RECENT_DEFAULT_DAYS = 3
RECENT_DEFAULT_LIMIT_PER_ACTOR = 40

TOPIC_TERMS = [
    '"Makkah Pact"',
    '"Mecca Pact"',
    '"Mecca Defence Pact"',
    '"Mecca Defense Pact"',
    '"Makkah Defence Pact"',
    '"Joint Defence Pact"',
]
TOPIC_CLAUSE = "(" + " OR ".join(TOPIC_TERMS) + ")"

ACTORS = {
    "Iran": ["Iran", "Iranian"],
    "Pakistan": ["Pakistan", "Pakistani"],
    "Saudi Arabia": ["Saudi Arabia", "Saudi"],
    "United States": ['"United States"', "U.S.", "USA"],
    "Turkiye": ["Turkiye", "Turkey", "Turkish"],
}

LIMIT_PER_ACTOR = 300
ACTOR_DELAY_MIN_SEC = 10
ACTOR_DELAY_MAX_SEC = 15

# twscrape fetches results ~20 at a time per underlying HTTP request (its
# GraphQL search page size) - there's no public hook to detect a page
# boundary directly, so we approximate it: pace every 20 tweets yielded.
REQUEST_PAGE_SIZE = 20
INTRA_SEARCH_DELAY_MIN_SEC = 2
INTRA_SEARCH_DELAY_MAX_SEC = 3

OUTPUT_COLUMNS = [
    "id", "matched_actor", "text", "username", "timestamp",
    "likes", "retweets", "hashtags",
]

HASHTAG_RE = re.compile(r"#\w+")


def extract_hashtags(text, api_hashtags):
    """twscrape's t.hashtags field misses hashtags in some tweets (visible
    #tags in the text weren't showing up), so pull them straight out of
    the tweet text with regex too and union with whatever the API gave us."""
    found = set(HASHTAG_RE.findall(text or ""))
    if api_hashtags:
        found.update("#" + h if not h.startswith("#") else h for h in api_hashtags)
    return ", ".join(sorted(found))


def build_query(actor_terms):
    actor_clause = "(" + " OR ".join(actor_terms) + ")"
    return f"{TOPIC_CLAUSE} {actor_clause} since:{SINCE_DATE}"


def load_existing():
    """Load whatever's already in x_data.csv (from a prior actor in this
    run, or a prior run) so we never lose progress."""
    if not os.path.exists(OUTPUT_CSV):
        return {}
    try:
        df = pd.read_csv(OUTPUT_CSV, dtype={"id": str})
    except Exception as exc:
        print(f"Could not read existing {OUTPUT_CSV} ({exc}); starting fresh.")
        return {}
    tweets_by_id = {}
    for _, row in df.iterrows():
        tweets_by_id[str(row["id"])] = {
            "id": str(row["id"]),
            "matched_actor": set(str(row.get("matched_actor", "")).split(", ")) - {""},
            "text": row.get("text", ""),
            "username": row.get("username", ""),
            "timestamp": row.get("timestamp", ""),
            "likes": row.get("likes", 0),
            "retweets": row.get("retweets", 0),
            "hashtags": row.get("hashtags", ""),
        }
    return tweets_by_id


def save(tweets_by_id):
    rows = []
    for data in tweets_by_id.values():
        row = dict(data)
        row["matched_actor"] = ", ".join(sorted(data["matched_actor"]))
        rows.append(row)
    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if not df.empty:
        # Rows reloaded from CSV carry "timestamp" as a plain string, while
        # freshly-fetched rows carry a real datetime object from twscrape -
        # sorting a column with mixed str/datetime values crashes. Coerce
        # everything to one consistent datetime dtype before sorting (this
        # also normalizes the format written back to the CSV).
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.sort_values("timestamp", ascending=False)
    df.to_csv(OUTPUT_CSV, index=False)


async def fetch_actor(api, actor, actor_terms):
    query = build_query(actor_terms)
    print(f"Searching for {actor}: {query}")
    tweets = []
    try:
        async for t in api.search(query, limit=LIMIT_PER_ACTOR):
            tweets.append(t)
            # Pace between (approximate) underlying page requests, not just
            # between actors - matters more at this higher per-actor volume.
            if len(tweets) % REQUEST_PAGE_SIZE == 0 and len(tweets) < LIMIT_PER_ACTOR:
                delay = random.uniform(INTRA_SEARCH_DELAY_MIN_SEC, INTRA_SEARCH_DELAY_MAX_SEC)
                await asyncio.sleep(delay)
    except Exception as exc:
        print(f"  [{actor}] search failed after {len(tweets)} tweets: {exc}")
    print(f"  -> {len(tweets)} tweets returned")
    return tweets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recent", action="store_true",
        help="Quick-refresh mode: only search the last --recent-days days, capped at "
             "--recent-limit tweets/actor. For 'what's new since last time', not a full backfill.",
    )
    parser.add_argument("--recent-days", type=int, default=RECENT_DEFAULT_DAYS)
    parser.add_argument("--recent-limit", type=int, default=RECENT_DEFAULT_LIMIT_PER_ACTOR)
    return parser.parse_args()


async def main():
    args = parse_args()
    global SINCE_DATE, LIMIT_PER_ACTOR
    if args.recent:
        SINCE_DATE = (dt.datetime.utcnow() - dt.timedelta(days=args.recent_days)).strftime("%Y-%m-%d")
        LIMIT_PER_ACTOR = args.recent_limit
        print(
            f"Recent mode: since={SINCE_DATE} ({args.recent_days} day(s) back), "
            f"limit={LIMIT_PER_ACTOR}/actor. Rate-limit delays are unchanged - "
            f"this is faster only because there's less to fetch.\n"
        )

    api = API()  # uses accounts.db in this folder (set up by setup_x_account.py)
    tweets_by_id = load_existing()
    n_before = len(tweets_by_id)
    print(f"Starting with {n_before} rows already in {OUTPUT_CSV}\n")

    actor_counts = {}
    actors = list(ACTORS.items())
    for i, (actor, actor_terms) in enumerate(actors):
        tweets = await fetch_actor(api, actor, actor_terms)
        actor_counts[actor] = len(tweets)

        for t in tweets:
            tid = str(t.id)
            entry = tweets_by_id.setdefault(tid, {
                "id": tid,
                "matched_actor": set(),
                "text": t.rawContent,
                "username": t.user.username if t.user else "",
                "timestamp": t.date,
                "likes": t.likeCount,
                "retweets": t.retweetCount,
                "hashtags": extract_hashtags(t.rawContent, t.hashtags),
            })
            entry["matched_actor"].add(actor)

        # Save after EVERY actor, not just at the end, so a failure partway
        # through (rate limit, account flag, crash) never loses progress.
        save(tweets_by_id)
        print(f"  Saved. {OUTPUT_CSV} now has {len(tweets_by_id)} unique rows total.\n")

        is_last = i == len(actors) - 1
        if not is_last:
            delay = random.uniform(ACTOR_DELAY_MIN_SEC, ACTOR_DELAY_MAX_SEC)
            print(f"Waiting {delay:.1f}s before the next actor ...\n")
            time.sleep(delay)

    n_after = len(tweets_by_id)
    print("=== Done ===")
    for actor, count in actor_counts.items():
        print(f"  {actor}: {count} tweets returned by search")
    print(f"Total unique rows in {OUTPUT_CSV}: {n_after}")
    print(f"NEW_ROWS_ADDED:{n_after - n_before}")


if __name__ == "__main__":
    asyncio.run(main())

"""
fetch_x_test.py

Phase 3 - small test ONLY. Confirms the cookies verified by
setup_x_account.py can actually log in and scrape: searches for the
keyword "Makkah Pact", pulls up to 10 tweets, prints them, and saves
them to x_test_sample.csv.

This is deliberately small and scoped to one keyword - do NOT scale
this up to the full actor/keyword list until this test is confirmed
working (run setup_x_account.py first).

2026-09-10: rewritten for the twscrape -> twifork/twikit pivot - see
x_auth.py's docstring and CLAUDE_INVESTIGATION_LOG.md in this folder.

Run:
    python fetch_x_test.py
"""

import asyncio
import re
import sys

import pandas as pd

from x_auth import build_client

QUERY = "Makkah Pact"
LIMIT = 10
OUTPUT_CSV = "x_test_sample.csv"

HASHTAG_RE = re.compile(r"#\w+")


def extract_hashtags(text, api_hashtags):
    """twikit's t.hashtags field can miss hashtags in some tweets, so pull
    them straight out of the visible text with regex too and union both."""
    found = set(HASHTAG_RE.findall(text or ""))
    if api_hashtags:
        found.update("#" + h if not h.startswith("#") else h for h in api_hashtags)
    return ", ".join(sorted(found))


async def main():
    client = build_client()

    print(f"Searching for \"{QUERY}\" (limit={LIMIT}) ...")
    try:
        tweets = await client.search_tweet(QUERY, "Latest", count=LIMIT)
    except Exception as exc:
        print(f"\nSearch failed: {exc}")
        print("Common causes: setup_x_account.py hasn't confirmed a working "
              "login yet, no internet access from wherever this is running, "
              "or the account is rate-limited/blocked.")
        sys.exit(1)
    print(f"-> {len(tweets)} tweets returned\n")

    rows = []
    for t in tweets:
        text = t.full_text
        row = {
            "text": text,
            "username": t.user.screen_name if t.user else "",
            "timestamp": t.created_at_datetime,
            "likes": t.favorite_count,
            "retweets": t.retweet_count,
            "hashtags": extract_hashtags(text, t.hashtags),
        }
        rows.append(row)
        print(
            f"[{row['timestamp']}] @{row['username']} "
            f"(likes={row['likes']}, rt={row['retweets']}): "
            f"{row['text'][:100]}"
        )

    df = pd.DataFrame(rows, columns=["text", "username", "timestamp", "likes", "retweets", "hashtags"])
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(df)} rows to {OUTPUT_CSV}")

    if len(df) == 0:
        print("No tweets came back - this could mean the login isn't "
              "actually valid (run setup_x_account.py first), the account "
              "got rate-limited/flagged, or there's genuinely no recent "
              "match for this keyword.")


if __name__ == "__main__":
    asyncio.run(main())

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

Requires: setup_x_account.py already confirmed the .env cookies are valid
(this script re-checks that itself too, before starting the actor loop).

Prints a "NEW_ROWS_ADDED:<n>" line at the end - tooling (e.g. the
dashboard) that wants the count of genuinely new rows this run added can
parse that line rather than re-diffing the CSV itself.

2026-09-10: rewritten for the twscrape -> twifork/twikit pivot - see
x_auth.py's docstring and CLAUDE_INVESTIGATION_LOG.md in this folder for
the full investigation behind this change. The actual query logic, actor
list, pacing, and output shape are UNCHANGED from before the pivot - only
the library doing the fetching (and therefore the exact API calls) changed.

2026-09-10 (later same day): the login check now retries once before
declaring the session dead, and a confirmed-good session's cookies are
persisted (both right after the check and again at the end of a
successful run) so the NEXT scheduled run starts from the freshest known
session state instead of always re-reading a static .env snapshot - see
x_auth.py's module docstring for the full "why" (X rotates cookies like
ct0 during normal use, and a transient Cloudflare-challenge blip can look
identical to a dead session). This does NOT eliminate manual cookie
refresh entirely - that's not possible against X as it currently stands
(also explained in x_auth.py) - it just reduces how often it's needed.

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
import sys
import time

import pandas as pd

from x_auth import build_client, is_logged_in_with_retry, save_session_cookies

# Preventive fix, applied pipeline-wide after a real incident: two other
# steps in this pipeline (preprocess_x_text.py, build_network_v2.py) each
# crashed with UnicodeEncodeError while printing scraped X/Twitter text
# (an emoji) to a Windows console defaulting to a single-byte codepage
# (cp1252) that can't represent it. This script prints each search query
# and per-actor error text, and processes the same scraped dataset on the
# same unattended schedule - reconfiguring stdout/stderr to UTF-8 with
# errors="replace" here too means an unprintable character is swapped for
# a placeholder instead of ever being able to crash this step the same way.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

OUTPUT_CSV = "x_data.csv"

# Durable per-run diagnostic trail, added 2026-09-08 while investigating a
# run of consecutive "SUCCESS - 0 new row(s)" results with no visibility
# into WHY: run_pipeline.py's run_step() only persists a step's FULL
# captured stdout+stderr to STEP_ERROR_LOG when that step's subprocess
# exits non-zero. fetch_actor() below catches every exception per actor
# internally and always returns normally (never re-raises), so this
# script always exits 0 - even if every single actor's search silently
# failed (an expired session, an account lock, a query error, etc.), the
# ONLY trace was a print() line that was never captured anywhere durable
# once the step "succeeded". This log closes that gap: every run appends
# one block here - actor-by-actor tweet counts AND any caught exception
# text - regardless of whether the run overall looked like a success, so a
# silently-degraded session leaves a permanent trace instead of looking
# identical to "topic genuinely has no new posts".
DIAG_LOG = "fetch_diagnostics.log"

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

# twikit's search_tweet() returns exactly REQUEST_PAGE_SIZE (or fewer, on
# the last page) tweets per call, with .next() making the next underlying
# HTTP request - a REAL page boundary (twscrape only let us approximate
# this, since it didn't expose one).
REQUEST_PAGE_SIZE = 20
INTRA_SEARCH_DELAY_MIN_SEC = 2
INTRA_SEARCH_DELAY_MAX_SEC = 3

OUTPUT_COLUMNS = [
    "id", "matched_actor", "text", "username", "timestamp",
    "likes", "retweets", "hashtags",
]

HASHTAG_RE = re.compile(r"#\w+")


def extract_hashtags(text, api_hashtags):
    """twikit's t.hashtags field can miss hashtags in some tweets (visible
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
        # freshly-fetched rows carry a real datetime object from twikit -
        # sorting a column with mixed str/datetime values crashes. Coerce
        # everything to one consistent datetime dtype before sorting (this
        # also normalizes the format written back to the CSV).
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.sort_values("timestamp", ascending=False)
    df.to_csv(OUTPUT_CSV, index=False)


async def fetch_actor(client, actor, actor_terms):
    query = build_query(actor_terms)
    print(f"Searching for {actor}: {query}")
    tweets = []
    error = None
    try:
        page = await client.search_tweet(query, "Latest", count=min(REQUEST_PAGE_SIZE, LIMIT_PER_ACTOR))
        # Stop on an empty page, not on a cursor - twikit hands back a
        # cursor even on an empty page (see Result's own docstring), so
        # looping on the cursor instead would never terminate.
        while page and len(tweets) < LIMIT_PER_ACTOR:
            for t in page:
                tweets.append(t)
                if len(tweets) >= LIMIT_PER_ACTOR:
                    break
            if len(tweets) >= LIMIT_PER_ACTOR:
                break
            # Pace between underlying page requests, not just between
            # actors - matters more at this higher per-actor volume.
            delay = random.uniform(INTRA_SEARCH_DELAY_MIN_SEC, INTRA_SEARCH_DELAY_MAX_SEC)
            await asyncio.sleep(delay)
            page = await page.next()
    except Exception as exc:
        # Deliberately still caught here (not re-raised) - one actor's search
        # failing (e.g. a transient rate limit) shouldn't abort the other 4
        # actors' searches. But the error text is now returned to the caller
        # (see `error` below) so it can be written to DIAG_LOG instead of
        # only ever reaching a print() that a "successful" run discards.
        error = str(exc)
        print(f"  [{actor}] search failed after {len(tweets)} tweets: {exc}")
    print(f"  -> {len(tweets)} tweets returned")
    return tweets, error


def append_diagnostics(mode_desc, since_date, limit_per_actor, actor_results, total_before, total_after):
    """Appends one block to DIAG_LOG for this run - every run, not just
    ones with a problem, so 'nothing unusual happened' is also on the
    record and a genuinely quiet period is distinguishable from a run this
    logging didn't cover yet. See DIAG_LOG's comment for the incident this
    was added for."""
    lines = [
        f"===== FETCH RUN {dt.datetime.utcnow().isoformat()}Z | mode={mode_desc} "
        f"since={since_date} limit={limit_per_actor}/actor ====="
    ]
    any_error = False
    for actor, info in actor_results.items():
        if info["error"]:
            any_error = True
            lines.append(f"  {actor}: {info['count']} tweet(s) returned [ERROR: {info['error']}]")
        else:
            lines.append(f"  {actor}: {info['count']} tweet(s) returned [ok]")
    lines.append(
        f"  rows before: {total_before}, rows after: {total_after}, "
        f"new: {total_after - total_before}{' - ONE OR MORE ACTORS ERRORED, see above' if any_error else ''}"
    )
    with open(DIAG_LOG, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n\n")


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

    # build_client() prefers a persisted session (x_session_cookies.json,
    # saved by a previous successful login) over the static .env snapshot,
    # so a cookie X rotated (e.g. ct0) since your last manual .env paste is
    # still in play - see x_auth.py's module docstring.
    client = build_client()
    print("Verifying the session is logged in before starting the actor loop ...")
    # is_logged_in_with_retry() retries once before reporting a dead
    # session - a single False can be the known, transient Cloudflare-
    # challenge-shell blip (see CLAUDE_INVESTIGATION_LOG.md), not proof the
    # cookies are actually bad. Only a second consecutive False here means
    # a genuinely dead session that needs a manual .env refresh.
    if not await is_logged_in_with_retry(client):
        print(
            "NOT logged in - X did not accept this session (checked twice). "
            "Run setup_x_account.py first to diagnose (expired/malformed "
            "cookies, or the known Cloudflare-challenge issue - see "
            "CLAUDE_INVESTIGATION_LOG.md). Stopping before wasting any "
            "actor searches on a session that won't work."
        )
        sys.exit(1)
    print("Logged in. Starting actor loop.\n")
    # Confirmed-good session - persist its current cookies now (may include
    # anything X rotated, e.g. ct0) so the NEXT run starts from this state
    # instead of the static .env snapshot. Best-effort; never raises.
    save_session_cookies(client)

    tweets_by_id = load_existing()
    n_before = len(tweets_by_id)
    print(f"Starting with {n_before} rows already in {OUTPUT_CSV}\n")

    actor_counts = {}
    actor_results = {}
    actors = list(ACTORS.items())
    for i, (actor, actor_terms) in enumerate(actors):
        tweets, error = await fetch_actor(client, actor, actor_terms)
        actor_counts[actor] = len(tweets)
        actor_results[actor] = {"count": len(tweets), "error": error}

        for t in tweets:
            tid = str(t.id)
            text = t.full_text
            entry = tweets_by_id.setdefault(tid, {
                "id": tid,
                "matched_actor": set(),
                "text": text,
                "username": t.user.screen_name if t.user else "",
                "timestamp": t.created_at_datetime,
                "likes": t.favorite_count,
                "retweets": t.retweet_count,
                "hashtags": extract_hashtags(text, t.hashtags),
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

    append_diagnostics(
        mode_desc="recent" if args.recent else "full",
        since_date=SINCE_DATE,
        limit_per_actor=LIMIT_PER_ACTOR,
        actor_results=actor_results,
        total_before=n_before,
        total_after=n_after,
    )

    # Second, end-of-run save: the actor loop can run for a long time (the
    # full historical pull especially) and make hundreds of requests, so
    # save once more here in case X rotated a cookie (e.g. ct0) partway
    # through, not just at the start. Best-effort; never raises.
    save_session_cookies(client)


if __name__ == "__main__":
    asyncio.run(main())

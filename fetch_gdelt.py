"""
fetch_gdelt.py

Phase 1 data collection script for the Makkah/Mecca Joint Defence Pact
project. Queries the GDELT 2.0 DOC API for news coverage that mentions the
pact (signed Aug 7, 2026) together with the five actor countries
(Iran, Pakistan, Saudi Arabia, United States, Turkiye), starting Aug 7, 2026,
and saves/merges the results into gdelt_data.csv.

Supplemental-run note (Aug 2026): the first full run only ever picked up
Saudi Arabia/Turkiye coverage through Aug 14 - Iran/Pakistan/United States
returned 0 rows that time (rate-limiting), and the window stopped short of
the X collection's Aug 17 end date. END_DATE below is now a FIXED cutoff
(Aug 17, not "whatever day you happen to run this") so a re-run lines up
exactly with the X data's date range. Nothing else changes: it's still the
same 2 grouped queries covering all 5 actors, and the existing merge-by-url
logic in load_existing()/main() already appends new rows and leaves
previously-collected rows (Aug 7-14, Saudi Arabia/Turkiye) untouched rather
than duplicating them - so one re-run safely retries the actors that failed
AND extends the date range for everyone, in a single pass.

Note on "tone": GDELT's DOC API only returns tone as an AGGREGATE across all
matching articles (via mode=tonechart / timelinetone), not as a field on each
individual article. So this script computes its own headline-level sentiment
score using VADER (a simple, well-established lexicon-based sentiment tool)
on each article's title. This is a proxy for tone, not GDELT's own number -
see the tone_score / tone_label columns.

Note on rate limiting: GDELT's free API is known to rate-limit (429) clients
that send requests without a normal browser User-Agent, and clients that fire
many requests back-to-back. This script sends a standard User-Agent, groups
the 5 actors into 2 queries instead of 5, and paces + backs off between
requests. If you've run this recently, GDELT's block can outlast a single
run - see the reminder printed at startup.

Run:
    python fetch_gdelt.py
"""

import os
import time
import datetime as dt

import requests
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
OUTPUT_CSV = "gdelt_data.csv"

# Standard browser User-Agent - GDELT is known to rate-limit requests
# that look like they're coming from a bare script/bot even at low volume.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}

# Pact was signed Aug 7, 2026 - collect from that date onward.
START_DATE = "20260807000000"
# Fixed end date (not "now") - pinned to Aug 17, 2026 to match the X
# collection window exactly, rather than drifting to today's date.
END_DATE = "20260817235959"

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

# Actors grouped into fewer queries to cut total request count (5 -> 2).
ACTOR_GROUPS = [
    ["Iran", "Pakistan"],
    ["Saudi Arabia", "United States", "Turkiye"],
]

MAX_RECORDS = 250
# Cooldown between the two grouped queries specifically. GDELT's rate-limit
# block has shown up on the second group even right after the first group
# succeeded, so this needs to be a real cooldown - not the same short pacing
# used for in-request retries below.
INTER_GROUP_COOLDOWN_SEC = 75  # between "Iran + Pakistan" and "Saudi Arabia + United States + Turkiye"
RETRY_BASE_DELAY_SEC = 3     # backoff for non-429 errors (attempt * this)
RATE_LIMIT_BACKOFF_SEC = 18  # backoff specifically after a 429, before retrying
MAX_RETRIES = 3
RECENT_RUN_WARNING_SEC = 20 * 60  # warn if last run was under 20 min ago

OUTPUT_COLUMNS = [
    "date", "actor", "actor_match_type", "title", "domain",
    "sourcecountry", "language", "url", "tone_score", "tone_label",
]

analyzer = SentimentIntensityAnalyzer()


def warn_if_run_recently():
    """GDELT's rate-limit block can outlast this script's own retry
    window, so nudge the user to space out runs."""
    print(
        "Reminder: if you've run this script in the last 20-30 minutes, "
        "GDELT may still be rate-limiting your IP from that run - "
        "consider waiting before continuing.\n"
    )
    if os.path.exists(OUTPUT_CSV):
        elapsed = time.time() - os.path.getmtime(OUTPUT_CSV)
        if elapsed < RECENT_RUN_WARNING_SEC:
            minutes_ago = elapsed / 60
            wait_more = (RECENT_RUN_WARNING_SEC - elapsed) / 60
            print(
                f"  -> {OUTPUT_CSV} was last updated {minutes_ago:.1f} min ago. "
                f"Consider waiting ~{wait_more:.0f} more minute(s) before running again.\n"
            )


def build_query(group_actors):
    all_terms = []
    for actor in group_actors:
        all_terms.extend(ACTORS[actor])
    actor_clause = "(" + " OR ".join(all_terms) + ")"
    return f"{TOPIC_CLAUSE} {actor_clause}"


def fetch_group_articles(group_label, group_actors):
    """Query the GDELT DOC API for one actor group, with retries."""
    query = build_query(group_actors)
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": MAX_RECORDS,
        "startdatetime": START_DATE,
        "enddatetime": END_DATE,
        "sort": "datedesc",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(GDELT_URL, params=params, headers=HEADERS, timeout=30)
            if resp.status_code == 429:
                wait = RATE_LIMIT_BACKOFF_SEC * attempt
                print(f"  [{group_label}] attempt {attempt}/{MAX_RETRIES} got 429 (rate limited) - waiting {wait}s before retry")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data.get("articles", [])
        except requests.exceptions.RequestException as exc:
            print(f"  [{group_label}] attempt {attempt}/{MAX_RETRIES} failed: {exc}")
            time.sleep(RETRY_BASE_DELAY_SEC * attempt)
        except ValueError:
            print(f"  [{group_label}] response was not valid JSON, skipping.")
            return []
    print(f"  [{group_label}] gave up after {MAX_RETRIES} attempts.")
    return []


def actors_in_title(title, group_actors):
    """Which actor(s) in this group are actually named in the headline."""
    title_lower = (title or "").lower()
    matched = []
    for actor in group_actors:
        for term in ACTORS[actor]:
            if term.strip('"').lower() in title_lower:
                matched.append(actor)
                break
    return matched


def score_tone(title):
    """Headline-level sentiment via VADER, used as a tone proxy."""
    if not title:
        return 0.0, "neutral"
    score = analyzer.polarity_scores(title)["compound"]
    if score >= 0.05:
        label = "positive"
    elif score <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return score, label


def load_existing():
    """Load previously collected rows so a partial/failed run never loses
    data that was already successfully gathered."""
    if not os.path.exists(OUTPUT_CSV):
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    try:
        existing = pd.read_csv(OUTPUT_CSV)
    except Exception as exc:
        print(f"Could not read existing {OUTPUT_CSV} ({exc}); starting fresh.")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    for col in OUTPUT_COLUMNS:
        if col not in existing.columns:
            existing[col] = "unknown" if col == "actor_match_type" else ""
    return existing[OUTPUT_COLUMNS]


def main():
    warn_if_run_recently()

    articles_by_url = {}  # url -> accumulated row data

    for i, group_actors in enumerate(ACTOR_GROUPS):
        group_label = " + ".join(group_actors)
        print(f"Fetching articles for group: {group_label} ...")
        articles = fetch_group_articles(group_label, group_actors)
        print(f"  -> {len(articles)} articles")

        for art in articles:
            url = art.get("url", "")
            if not url:
                continue
            title = art.get("title", "")
            matched = actors_in_title(title, group_actors)
            match_type = "headline_match"
            if not matched:
                # Article matched the group's combined query, but no
                # single actor name was found in the headline itself -
                # fall back to tagging the whole group as a lower-
                # confidence match.
                matched = list(group_actors)
                match_type = "group_level"

            entry = articles_by_url.setdefault(url, {
                "date": art.get("seendate", ""),
                "title": title,
                "domain": art.get("domain", ""),
                "sourcecountry": art.get("sourcecountry", ""),
                "language": art.get("language", ""),
                "actors": set(),
                "match_types": set(),
            })
            entry["actors"].update(matched)
            entry["match_types"].add(match_type)

        if i < len(ACTOR_GROUPS) - 1:
            print(
                f"  Cooling down {INTER_GROUP_COOLDOWN_SEC}s before the next query group "
                f"(GDELT's block has shown up on the second group even right after the "
                f"first one succeeded, so this needs a real gap, not just a short pause)..."
            )
            time.sleep(INTER_GROUP_COOLDOWN_SEC)

    new_rows = []
    for url, data in articles_by_url.items():
        tone_score, tone_label = score_tone(data["title"])
        # If any headline match happened for this article, prefer that
        # label; otherwise mark it group_level.
        match_type = "headline_match" if "headline_match" in data["match_types"] else "group_level"
        new_rows.append({
            "date": data["date"],
            "actor": ", ".join(sorted(data["actors"])),
            "actor_match_type": match_type,
            "title": data["title"],
            "domain": data["domain"],
            "sourcecountry": data["sourcecountry"],
            "language": data["language"],
            "url": url,
            "tone_score": tone_score,
            "tone_label": tone_label,
        })

    new_df = pd.DataFrame(new_rows, columns=OUTPUT_COLUMNS)
    if not new_df.empty:
        new_df["date"] = pd.to_datetime(
            new_df["date"], format="%Y%m%dT%H%M%SZ", errors="coerce"
        )

    existing_df = load_existing()
    if not existing_df.empty:
        existing_df["date"] = pd.to_datetime(existing_df["date"], errors="coerce")

    print(f"Existing rows already in {OUTPUT_CSV}: {len(existing_df)}")
    print(f"New rows fetched this run: {len(new_df)}")

    combined = pd.concat([existing_df, new_df], ignore_index=True)
    if combined.empty:
        combined = pd.DataFrame(columns=OUTPUT_COLUMNS)
    else:
        # Keep the freshest data per URL (new run overrides old on conflict),
        # but never drop a URL that only exists in one of the two sets.
        combined = combined.drop_duplicates(subset="url", keep="last")
        combined = combined.sort_values("date", ascending=False)

    combined.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(combined)} total rows to {OUTPUT_CSV} "
          f"({len(existing_df)} carried over + {len(new_df)} from this run, deduplicated by url)")


if __name__ == "__main__":
    main()

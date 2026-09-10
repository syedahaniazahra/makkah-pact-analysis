"""
build_daily_timelines.py

Phase 7, steps 2-3 - daily X timeline, daily GDELT timeline, and a
GDELT-vs-X cross-correlation check.

--- Step 2: X timeline (English rows only, x_data_cleaned.csv) ---
Groups by calendar date (UTC). "total_posts" is every English row that
day. Per-actor counts use "actors_mentioned" (a post naming 3 actors adds
1 to each of those 3 actor columns that day - these are mention counts,
not row counts, so they don't have to sum to total_posts).
Output: daily_actor_timeline.csv
Spikes are flagged as any date where total_posts > mean + 1 stdev across
the collection window (a plain, explainable rule - not tuned).

--- Step 3: GDELT timeline (gdelt_data.csv) ---
Same grouping logic, applied to GDELT. The "actor" field there can hold
combos like "Saudi Arabia, Turkiye" (both keywords matched the article),
so it's split on "," the same way actors_mentioned would be.
Output: daily_gdelt_timeline.csv

Cross-correlation: for each of the 5 actors, if GDELT has ANY coverage of
that actor, align GDELT's daily article count with X's daily mention
count for that actor on (a) the same date and (b) X's count the
following day (does GDELT coverage lead X mentions by a day?). Pearson r
is computed only when there are >= 5 overlapping dates AND both series
have non-zero variance; below that, or when GDELT has zero coverage of
an actor at all, the result is reported in words instead of forcing a
number on too few points.

Run:
    python build_daily_timelines.py             # X + GDELT + cross-correlation
    python build_daily_timelines.py --x-only     # X timeline only, GDELT untouched
                                                  # (used by the dashboard's quick
                                                  # refresh, which doesn't re-fetch
                                                  # GDELT so there's nothing new to
                                                  # cross-correlate)
"""

import argparse
import ast
import sys

import numpy as np
import pandas as pd

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

X_CSV = "x_data_cleaned.csv"
GDELT_CSV = "gdelt_data.csv"
X_TIMELINE_CSV = "daily_actor_timeline.csv"
GDELT_TIMELINE_CSV = "daily_gdelt_timeline.csv"

ACTORS = ["Iran", "Pakistan", "Saudi Arabia", "United States", "Turkiye"]
MIN_POINTS_FOR_CORRELATION = 5


def require_columns(df, columns, upstream_script, csv_name):
    """See extract_actor_mentions.py's require_columns() for the full
    rationale - same convention, same SKIP_REASON + exit(3) contract that
    run_pipeline.py's run_step() recognizes as a graceful skip."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        print(
            f"SKIP_REASON: column(s) {missing} not found in {csv_name} - "
            f"{upstream_script} has not completed successfully yet this cycle. "
            f"Nothing to do until it does; will retry next cycle."
        )
        sys.exit(3)


def build_x_timeline():
    df = pd.read_csv(X_CSV, dtype={"id": str})
    require_columns(df, ["actors_mentioned"], "extract_actor_mentions.py", X_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    en = df[df["detected_language"] == "en"].copy()
    en["date"] = en["timestamp"].dt.date

    n_dropped = en["date"].isna().sum()
    if n_dropped:
        print(f"Note: {n_dropped} English row(s) had an unparseable timestamp, excluded from the timeline")
    en = en.dropna(subset=["date"])

    en["_actor_set"] = en["actors_mentioned"].apply(
        lambda s: set(ast.literal_eval(s)) if isinstance(s, str) and s else set()
    )

    total_by_date = en.groupby("date").size().rename("total_posts")

    actor_rows = []
    for _, r in en.iterrows():
        for a in r["_actor_set"]:
            actor_rows.append({"date": r["date"], "actor": a})
    actor_long = pd.DataFrame(actor_rows, columns=["date", "actor"])
    actor_wide = (
        actor_long.groupby(["date", "actor"]).size().unstack(fill_value=0)
        if not actor_long.empty
        else pd.DataFrame(columns=ACTORS)
    )
    for a in ACTORS:
        if a not in actor_wide.columns:
            actor_wide[a] = 0
    actor_wide = actor_wide[ACTORS]

    timeline = pd.concat([total_by_date, actor_wide], axis=1).fillna(0).reset_index()
    timeline = timeline.rename(columns={"index": "date"})
    for a in ACTORS:
        timeline[a] = timeline[a].astype(int)
    timeline["total_posts"] = timeline["total_posts"].astype(int)
    timeline = timeline.sort_values("date")

    mean_posts = timeline["total_posts"].mean()
    std_posts = timeline["total_posts"].std()
    spike_threshold = mean_posts + std_posts
    timeline["is_spike"] = timeline["total_posts"] > spike_threshold

    timeline.to_csv(X_TIMELINE_CSV, index=False)
    print(f"Saved -> {X_TIMELINE_CSV} ({len(timeline)} dates)")
    print(f"Spike threshold (mean + 1 stdev of total_posts): {spike_threshold:.1f}")
    spikes = timeline[timeline["is_spike"]]
    print("Spike dates:")
    print(spikes[["date", "total_posts"]].to_string(index=False))

    return timeline


def split_gdelt_actors(actor_field):
    if not isinstance(actor_field, str) or not actor_field.strip():
        return []
    return [a.strip() for a in actor_field.split(",") if a.strip()]


def build_gdelt_timeline():
    g = pd.read_csv(GDELT_CSV)
    if g.empty:
        print("gdelt_data.csv is empty (no rows) - skipping GDELT timeline")
        return pd.DataFrame(columns=["date", "total_articles"] + ACTORS)

    g["date"] = pd.to_datetime(g["date"], errors="coerce").dt.date
    g = g.dropna(subset=["date"])

    total_by_date = g.groupby("date").size().rename("total_articles")

    actor_rows = []
    for _, r in g.iterrows():
        for a in split_gdelt_actors(r["actor"]):
            actor_rows.append({"date": r["date"], "actor": a})
    actor_long = pd.DataFrame(actor_rows, columns=["date", "actor"])
    actor_wide = (
        actor_long.groupby(["date", "actor"]).size().unstack(fill_value=0)
        if not actor_long.empty
        else pd.DataFrame(columns=ACTORS)
    )
    for a in ACTORS:
        if a not in actor_wide.columns:
            actor_wide[a] = 0
    actor_wide = actor_wide[ACTORS]

    timeline = pd.concat([total_by_date, actor_wide], axis=1).fillna(0).reset_index()
    timeline = timeline.rename(columns={"index": "date"})
    for a in ACTORS:
        timeline[a] = timeline[a].astype(int)
    timeline["total_articles"] = timeline["total_articles"].astype(int)
    timeline = timeline.sort_values("date")

    timeline.to_csv(GDELT_TIMELINE_CSV, index=False)
    print(f"\nSaved -> {GDELT_TIMELINE_CSV} ({len(timeline)} dates)")
    covered_actors = [a for a in ACTORS if timeline[a].sum() > 0]
    print(f"Actors with any GDELT coverage: {covered_actors}")

    return timeline


def pearson_or_none(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < MIN_POINTS_FOR_CORRELATION:
        return None, f"only {len(x)} overlapping date(s) - too few to compute a meaningful correlation"
    if np.std(x) == 0 or np.std(y) == 0:
        return None, "one of the two series is constant (no variance) over this window - correlation is undefined"
    r = float(np.corrcoef(x, y)[0, 1])
    return r, None


def cross_correlate(x_timeline, gdelt_timeline):
    print("\n=== GDELT vs X cross-correlation, per actor ===")
    x_by_date = x_timeline.set_index("date")
    g_by_date = gdelt_timeline.set_index("date")

    for actor in ACTORS:
        if actor not in g_by_date.columns or g_by_date[actor].sum() == 0:
            print(f"{actor}: no GDELT coverage at all in this dataset - cannot assess correlation")
            continue

        # Use every date in the GDELT collection window (including days with
        # 0 articles for this actor) - a proper aligned daily series, not
        # just the days GDELT happened to mention the actor.
        g_dates = sorted(g_by_date.index)
        g_series = g_by_date.loc[g_dates, actor]

        # same-day
        same_day_x = [x_by_date[actor].get(d, 0) for d in g_dates]
        r_same, note_same = pearson_or_none(g_series.values, same_day_x)

        # next-day (GDELT date d vs X mentions on d+1)
        next_day_dates = [pd.Timestamp(d) + pd.Timedelta(days=1) for d in g_dates]
        next_day_dates = [d.date() for d in next_day_dates]
        next_day_x = [x_by_date[actor].get(d, 0) for d in next_day_dates]
        r_next, note_next = pearson_or_none(g_series.values, next_day_x)

        print(f"\n{actor} (GDELT window: {g_dates[0]} to {g_dates[-1]}, {len(g_dates)} days):")
        if r_same is not None:
            print(f"  same-day Pearson r (GDELT articles vs X mentions): {r_same:.2f}")
        else:
            print(f"  same-day: {note_same}")
        if r_next is not None:
            print(f"  next-day Pearson r (GDELT articles vs X mentions the following day): {r_next:.2f}")
        else:
            print(f"  next-day: {note_next}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--x-only", action="store_true",
        help="Only rebuild daily_actor_timeline.csv - skip GDELT entirely (used by "
             "the dashboard's quick refresh, which doesn't touch GDELT data).",
    )
    args = parser.parse_args()

    x_timeline = build_x_timeline()
    if args.x_only:
        print("\n--x-only: skipping GDELT timeline and cross-correlation.")
        return
    gdelt_timeline = build_gdelt_timeline()
    if not gdelt_timeline.empty:
        cross_correlate(x_timeline, gdelt_timeline)


if __name__ == "__main__":
    main()

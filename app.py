"""
app.py

Phase 8 - Streamlit dashboard for the Makkah/Mecca Joint Defence Pact
project. Reads the already-built analysis outputs (x_data_cleaned.csv,
actor_network.json, daily_actor_timeline.csv, hashtag_frequency.csv, and
daily_gdelt_timeline.csv if present) - this file does no new data
processing beyond what's needed to render charts; all the heavy lifting
(cleaning, dedup, sentiment, etc.) already happened in the earlier phase
scripts.

Layout: a header (title + last-updated timestamp + a Refresh Data button)
followed by five tabs - Actor Network, Timeline, Sentiment & Engagement,
Hashtags, and a Data Explorer table.

Run:
    streamlit run app.py
"""

import ast
import os
import re
import subprocess
import sys
import time
from datetime import datetime

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

X_DATA_CSV = os.path.join(DATA_DIR, "x_data_cleaned.csv")
NETWORK_JSON = os.path.join(DATA_DIR, "actor_network.json")
TIMELINE_CSV = os.path.join(DATA_DIR, "daily_actor_timeline.csv")
GDELT_TIMELINE_CSV = os.path.join(DATA_DIR, "daily_gdelt_timeline.csv")
HASHTAG_FREQ_CSV = os.path.join(DATA_DIR, "hashtag_frequency.csv")
ACTOR_FREQ_DEDUP_CSV = os.path.join(DATA_DIR, "actor_frequency_deduped.csv")
ACTOR_COOCCUR_DEDUP_CSV = os.path.join(DATA_DIR, "actor_cooccurrence_deduped.csv")

# The full quick-refresh chain now runs as an independent background
# process (refresh_worker.py) instead of inline here - see the "Refresh
# Data pipeline" section below for why (non-blocking + no timeout).
FETCH_X_SCRIPT = os.path.join(DATA_DIR, "fetch_x_data.py")
CLEAN_SCRIPT = os.path.join(DATA_DIR, "clean_x_data.py")
PREPROCESS_SCRIPT = os.path.join(DATA_DIR, "preprocess_x_text.py")
HASHTAGS_SCRIPT = os.path.join(DATA_DIR, "extract_topics_hashtags.py")
ACTOR_MENTIONS_SCRIPT = os.path.join(DATA_DIR, "extract_actor_mentions.py")
SENTIMENT_SCRIPT = os.path.join(DATA_DIR, "add_sentiment.py")
DEDUPE_SCRIPT = os.path.join(DATA_DIR, "dedupe_actor_cooccurrence.py")
NETWORK_SCRIPT = os.path.join(DATA_DIR, "build_actor_network.py")
TIMELINE_SCRIPT = os.path.join(DATA_DIR, "build_daily_timelines.py")
REFRESH_WORKER_SCRIPT = os.path.join(DATA_DIR, "refresh_worker.py")
REFRESH_LOG = os.path.join(DATA_DIR, "refresh_progress.log")
REFRESH_LOCK = os.path.join(DATA_DIR, "refresh.lock")

DONE_RE = re.compile(r"REFRESH_DONE:(success|nochange|error):(.*)", re.DOTALL)

ACTORS = ["Iran", "Pakistan", "Saudi Arabia", "United States", "Turkiye"]
# A couple of stray pre-pact outlier posts (e.g. a 2023 tweet, an old retweet
# surfaced by keyword search) can appear in the raw data with dates far
# before the pact existed. They're kept in the underlying CSVs (no rows are
# dropped from the data itself), but a timeline chart spanning "2023 to now"
# would crush the actual Aug 7-17 window into a sliver - so charts default to
# this floor, with a caption noting anything excluded from the view.
CHART_MIN_DATE = pd.Timestamp("2026-08-01").date()
# The pact's actual signing date is a fixed historical fact, not something
# that changes as more data comes in - unlike spike dates (see the Timeline
# tab below), which are now read live from daily_actor_timeline.csv's
# is_spike column instead of being hardcoded here, so they stay correct as
# the collection window grows past whatever day this was last edited.
SIGNING_DATE = pd.Timestamp("2026-08-07").date()

st.set_page_config(page_title="Makkah Pact Monitor", layout="wide")


# --------------------------------------------------------------------------
# Data loading - each loader is cached but keyed on the source file's mtime,
# so a refreshed file (via the button, or a manually re-run script) busts
# the cache automatically without needing a hard restart.
# --------------------------------------------------------------------------

def _safe_list(value):
    """Parse a stringified list column (e.g. "['Iran', 'Pakistan']"); blank/
    NaN for non-English rows becomes an empty list rather than an error."""
    if isinstance(value, str) and value.strip():
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []
    return []


# Columns added by a LATER stage of the refresh pipeline than clean_x_data.py
# (which rewrites x_data_cleaned.csv from scratch first). If this file is
# read while a refresh is mid-chain - e.g. a second browser tab open during
# someone else's refresh - these can be transiently absent. Defaulting them
# here means a mid-refresh read shows stale-looking-but-safe data instead of
# crashing the whole app; the pipeline's own st.stop()-on-failure guard is
# still what actually prevents a BROKEN (vs. merely in-progress) file from
# ever being treated as final.
OPTIONAL_STAGE_COLUMNS = {
    "confirmed_flag": None, "clean_text": "", "clean_text_lemmatized": "",
    "long_form_excerpt": "", "hashtags_extracted": "", "actor_match_diff": "",
    "actor_match_verified": None, "actors_mentioned": "",
    "sentiment_score": None, "sentiment_label": "",
}


@st.cache_data
def _load_x_data(mtime):
    df = pd.read_csv(X_DATA_CSV, dtype={"id": str})
    for col, default in OPTIONAL_STAGE_COLUMNS.items():
        if col not in df.columns:
            df[col] = default
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["date"] = df["timestamp"].dt.date
    df["actors_list"] = df["actors_mentioned"].apply(_safe_list)
    df["likes"] = pd.to_numeric(df["likes"], errors="coerce")
    df["retweets"] = pd.to_numeric(df["retweets"], errors="coerce")
    df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")
    return df


def load_x_data():
    mtime = os.path.getmtime(X_DATA_CSV)
    return _load_x_data(mtime), mtime


@st.cache_data
def load_network(mtime):
    import json
    with open(NETWORK_JSON) as f:
        return json.load(f)


@st.cache_data
def load_timeline(mtime):
    df = pd.read_csv(TIMELINE_CSV, parse_dates=["date"])
    df["date"] = df["date"].dt.date
    return df


@st.cache_data
def load_gdelt_timeline(mtime):
    df = pd.read_csv(GDELT_TIMELINE_CSV, parse_dates=["date"])
    df["date"] = df["date"].dt.date
    return df


@st.cache_data
def load_hashtag_freq(mtime):
    return pd.read_csv(HASHTAG_FREQ_CSV)


# --------------------------------------------------------------------------
# Refresh Data pipeline - NON-BLOCKING.
#
# The whole fetch->clean->preprocess->...->timeline chain used to run
# SYNCHRONOUSLY inside this button's click handler (subprocess.run(),
# which blocks until each step finishes). That froze the entire page for
# however long the chain took, and a fixed 900s timeout on top of that
# could cut off a legitimately slow step (the X fetch has deliberate
# rate-limit delays) - which is what caused a real refresh to time out.
#
# Fix: the button now launches refresh_worker.py as an independent
# background process via subprocess.Popen() (does NOT wait for it) and
# immediately returns control to Streamlit. All status after that comes
# from two plain files refresh_worker.py writes:
#   refresh.lock         - exists while a refresh is in progress (deleted
#                           by the worker when it finishes, success or not)
#   refresh_progress.log - live progress text, appended to as each step
#                           runs; ends with one REFRESH_DONE:... line
# Reading files instead of relying on Streamlit session state means "is a
# refresh running" is visible correctly even from a fresh page load or a
# different browser tab - not just the tab that clicked the button.
#
# The polling loop below (time.sleep + st.rerun) reruns this script every
# ~2.5s while a refresh is in progress. Each rerun only costs ~2.5s of
# wait, not the length of the whole pipeline - so the page keeps repainting
# and stays interactive between polls, instead of hanging on one giant
# blocking call. The actual heavy work happens entirely in the separate
# refresh_worker.py OS process, which has no timeout at all.
# --------------------------------------------------------------------------

def _read_text_if_exists(path):
    if os.path.exists(path):
        with open(path) as f:
            return f.read()
    return None


def _launch_refresh():
    """Start refresh_worker.py in the background and return immediately -
    does not wait for it. `-u` keeps the worker's own prints unbuffered so
    they reach the log file promptly instead of sitting in a block buffer."""
    with open(REFRESH_LOG, "w"):
        pass  # truncate any old log from a previous run
    log_fh = open(REFRESH_LOG, "a")
    subprocess.Popen(
        [sys.executable, "-u", REFRESH_WORKER_SCRIPT],
        cwd=DATA_DIR,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )


def _refresh_status():
    """('running' | 'done' | 'idle', log_text, done_status, done_detail).
    Driven entirely by the lock file + log file, not session state, so it
    reads correctly from any tab/session, not just the one that clicked
    Refresh Data."""
    log_text = _read_text_if_exists(REFRESH_LOG) or ""
    m = DONE_RE.search(log_text)
    if m:
        return "done", log_text, m.group(1), m.group(2).strip()
    if os.path.exists(REFRESH_LOCK):
        return "running", log_text, None, None
    return "idle", log_text, None, None


# --------------------------------------------------------------------------
# Header: title, last-updated timestamp, Refresh Data button
# --------------------------------------------------------------------------

st.title("Makkah / Mecca Joint Defence Pact — Social & News Monitor")

refresh_status, refresh_log_text, done_status, done_detail = _refresh_status()
if refresh_status == "done":
    # Clear the cache (and consume the log) BEFORE loading data below, so
    # this same render already shows fresh data next to the completion
    # banner - no extra rerun/flicker needed.
    st.cache_data.clear()
    try:
        os.remove(REFRESH_LOG)
    except OSError:
        pass

df, x_mtime = load_x_data()
en_df = df[df["detected_language"] == "en"].copy()
# The 12 manually-confirmed slur/harassment rows are excluded from display
# everywhere in this dashboard, not just the explorer table - per the
# earlier decision not to quote them directly.
en_df_safe = en_df[en_df["confirmed_flag"] != True]  # noqa: E712 (keep NaN/False, drop True)

# --------------------------------------------------------------------------
# Data-completeness check.
#
# clean_x_data.py rebuilds x_data_cleaned.csv from x_data.csv FROM SCRATCH
# every time it runs, and only adds the columns IT computes (language,
# clustering, flags). Everything the Sentiment tab and the Data Explorer
# filters depend on - sentiment_score, actors_mentioned, sentiment_label -
# is added by LATER scripts in the chain (preprocess_x_text.py ->
# extract_topics_hashtags.py -> extract_actor_mentions.py ->
# add_sentiment.py). If clean_x_data.py is ever run by itself (e.g. by
# hand, outside the dashboard) without the rest of the chain following it,
# x_data_cleaned.csv is left with those columns present-but-empty (thanks
# to the defensive defaulting in _load_x_data below, which exists so a
# refresh caught mid-flight doesn't crash the page) - which silently looks
# like "the Sentiment tab has no data" and "every Data Explorer filter
# returns 0 rows", with no error anywhere to explain why. This check
# catches that state explicitly instead of leaving it to look like a bug.
_en_total = len(en_df_safe)
if _en_total:
    _sentiment_missing = int(en_df_safe["sentiment_score"].isna().sum())
    _actors_missing = int((en_df_safe["actors_mentioned"].fillna("").astype(str).str.strip() == "").sum())
    _incomplete = max(_sentiment_missing, _actors_missing)
    if _incomplete == _en_total:
        st.error(
            "⚠️ **This data hasn't finished processing.** None of the "
            f"{_en_total} English post(s) have sentiment or actor-mention "
            "analysis yet - the Sentiment & Engagement tab will be empty and "
            "every Data Explorer filter will return 0 results until this is "
            "fixed. This happens when `clean_x_data.py` was run by itself "
            "without the rest of the chain after it (it rebuilds "
            "x_data_cleaned.csv from scratch, which clears those columns "
            "until preprocess/hashtags/actor-mentions/sentiment re-add them). "
            "**Fix:** click 🔄 Refresh Data above (it runs the full chain in "
            "order), or if you're running scripts by hand, run them in this "
            "order after clean_x_data.py: preprocess_x_text.py -> "
            "extract_topics_hashtags.py -> extract_actor_mentions.py -> "
            "add_sentiment.py."
        )
    elif _incomplete > 0:
        st.warning(
            f"⚠️ {_incomplete} of {_en_total} English post(s) are missing "
            "sentiment/actor-mention analysis (likely rows added since the "
            "last full pipeline run). Charts below reflect only the "
            f"{_en_total - _incomplete} fully-processed post(s) - click 🔄 "
            "Refresh Data above to catch the rest up."
        )

header_left, header_right = st.columns([3, 1])
with header_left:
    last_updated = datetime.fromtimestamp(x_mtime).strftime("%Y-%m-%d %H:%M:%S")
    st.caption(f"Data last updated: {last_updated}")
with header_right:
    st.caption(
        "⚠️ Fetches only the last few days (small, fast pull), then re-cleans/"
        "re-scores the pipeline in the background - the page stays usable "
        "while it runs."
    )
    if refresh_status == "running":
        st.button("🔄 Refreshing…", disabled=True, width='stretch')
    else:
        if st.button("🔄 Refresh Data", width='stretch'):
            missing_scripts = [
                p for p in [
                    REFRESH_WORKER_SCRIPT, FETCH_X_SCRIPT, CLEAN_SCRIPT, PREPROCESS_SCRIPT,
                    HASHTAGS_SCRIPT, ACTOR_MENTIONS_SCRIPT, SENTIMENT_SCRIPT, DEDUPE_SCRIPT,
                    NETWORK_SCRIPT, TIMELINE_SCRIPT,
                ] if not os.path.exists(p)
            ]
            if missing_scripts:
                st.error("Can't find: " + ", ".join(os.path.basename(p) for p in missing_scripts))
            else:
                _launch_refresh()
                st.rerun()

if refresh_status == "running":
    st.info(
        "🔄 Refresh running in the background — this page updates automatically "
        "every ~2-3s. Everything below is still the last-loaded data - browse "
        "away, it'll swap in as soon as the refresh finishes."
    )
    with st.expander("Live progress log", expanded=True):
        st.code(refresh_log_text[-3000:] or "Starting…", language=None)
    # NOTE: the actual "wait then poll again" call is at the very BOTTOM of
    # this file, after every tab has rendered - not here. st.rerun() halts
    # the script immediately, so if it were called at this point (before the
    # tabs below), the tabs would never even run and the whole rest of the
    # page would just vanish while a refresh was in progress - which is
    # exactly the bug this comment is here to prevent reintroducing.

elif refresh_status == "done":
    # Cache was already cleared and the log already removed above (before
    # df was loaded), so the charts below are already showing fresh data
    # in this same render - this banner just reports what happened.
    if done_status == "success":
        st.success(f"Refresh complete — {done_detail}")
    elif done_status == "nochange":
        st.info(f"Refresh finished — {done_detail}")
    else:
        st.error(f"Refresh failed — {done_detail}")
        with st.expander("Full log"):
            st.code(refresh_log_text[-4000:], language=None)

st.divider()

tab_network, tab_timeline, tab_sentiment, tab_hashtags, tab_explorer = st.tabs(
    ["Actor Network", "Timeline", "Sentiment & Engagement", "Hashtags", "Data Explorer"]
)


# --------------------------------------------------------------------------
# Tab 1: Actor network graph
# --------------------------------------------------------------------------

with tab_network:
    st.subheader("Actor co-mention network")
    st.caption(
        "Node size = deduplicated mention count (each near-duplicate/repost cluster "
        "counted once). Edge thickness and color = deduplicated co-occurrence count "
        "between that pair of actors."
    )

    if not os.path.exists(NETWORK_JSON):
        st.warning(f"{os.path.basename(NETWORK_JSON)} not found.")
    else:
        network = load_network(os.path.getmtime(NETWORK_JSON))
        nodes = network["nodes"]
        edges = network["edges"]

        G = nx.Graph()
        for n in nodes:
            G.add_node(n["id"], size=n["size"])
        for e in edges:
            if e["weight"] > 0:
                G.add_edge(e["source"], e["target"], weight=e["weight"])

        pos = nx.spring_layout(G, weight="weight", seed=42, k=1.1)

        max_weight = max((e["weight"] for e in edges), default=1) or 1
        edge_traces = []
        for e in edges:
            if e["weight"] <= 0:
                continue
            x0, y0 = pos[e["source"]]
            x1, y1 = pos[e["target"]]
            width = 1 + 7 * (e["weight"] / max_weight)
            edge_traces.append(
                go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode="lines",
                    line=dict(width=width, color="rgba(120,120,180,0.55)"),
                    hoverinfo="text",
                    text=f"{e['source']} – {e['target']}: {e['weight']}",
                    showlegend=False,
                )
            )

        sizes = [n["size"] for n in nodes]
        min_size, max_size = min(sizes), max(sizes)

        def scale_node_size(v):
            if max_size == min_size:
                return 50
            return 30 + 55 * (v - min_size) / (max_size - min_size)

        node_trace = go.Scatter(
            x=[pos[n["id"]][0] for n in nodes],
            y=[pos[n["id"]][1] for n in nodes],
            mode="markers+text",
            text=[n["label"] for n in nodes],
            textposition="top center",
            hovertext=[f"{n['label']}: {n['size']} mentions" for n in nodes],
            hoverinfo="text",
            marker=dict(
                size=[scale_node_size(n["size"]) for n in nodes],
                color=sizes,
                colorscale="Blues",
                showscale=False,
                line=dict(width=1.5, color="white"),
            ),
            showlegend=False,
        )

        fig = go.Figure(data=edge_traces + [node_trace])
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=520,
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="white",
        )
        st.plotly_chart(fig, width='stretch')

        with st.expander("Edge weights (all 10 pairs)"):
            edge_df = pd.DataFrame(edges).sort_values("weight", ascending=False)
            st.dataframe(edge_df, width='stretch', hide_index=True)


# --------------------------------------------------------------------------
# Tab 2: Timeline
# --------------------------------------------------------------------------

with tab_timeline:
    st.subheader("Daily post volume per actor")

    if not os.path.exists(TIMELINE_CSV):
        st.warning(f"{os.path.basename(TIMELINE_CSV)} not found.")
    else:
        timeline_full = load_timeline(os.path.getmtime(TIMELINE_CSV))
        timeline = timeline_full[timeline_full["date"] >= CHART_MIN_DATE]
        n_excluded = len(timeline_full) - len(timeline)

        selected_actors = st.multiselect(
            "Actors to show", ACTORS, default=ACTORS, key="timeline_actors"
        )
        show_total = st.checkbox("Show total posts/day", value=False)

        gdelt_available = os.path.exists(GDELT_TIMELINE_CSV)
        show_gdelt = False
        if gdelt_available:
            show_gdelt = st.checkbox(
                "Overlay GDELT article volume (secondary source, right axis)",
                value=False,
                help="GDELT coverage in this dataset may not cover all 5 actors or "
                     "the full date range - see the Timeline note below the chart.",
            )

        fig = go.Figure()
        for actor in selected_actors:
            if actor in timeline.columns:
                fig.add_trace(go.Scatter(
                    x=timeline["date"], y=timeline[actor],
                    mode="lines+markers", name=actor,
                ))
        if show_total:
            fig.add_trace(go.Scatter(
                x=timeline["date"], y=timeline["total_posts"],
                mode="lines+markers", name="Total posts",
                line=dict(color="black", dash="dot"),
            ))

        if show_gdelt and gdelt_available:
            gdelt_timeline = load_gdelt_timeline(os.path.getmtime(GDELT_TIMELINE_CSV))
            fig.add_trace(go.Scatter(
                x=gdelt_timeline["date"], y=gdelt_timeline["total_articles"],
                mode="lines+markers", name="GDELT articles/day",
                line=dict(color="gray"), yaxis="y2",
            ))
            fig.update_layout(
                yaxis2=dict(title="GDELT articles", overlaying="y", side="right")
            )

        # Spike dates come live from daily_actor_timeline.csv's own is_spike
        # column (computed in build_daily_timelines.py as total_posts > mean
        # + 1 stdev over the actual collection window) - NOT hardcoded here.
        # That column recomputes every time the pipeline runs, so as the
        # collection window grows, the annotated spikes always reflect the
        # current data instead of going stale (this used to be a fixed
        # {Aug 14, Aug 17} dict that silently stopped matching reality once
        # more days of data came in).
        spike_dates = sorted(timeline.loc[timeline["is_spike"], "date"]) if "is_spike" in timeline.columns else []
        latest_data_date = timeline["date"].max() if not timeline.empty else None

        # plotly's add_vline chokes on bare datetime.date objects (it tries
        # to average the x-position numerically) - pass ISO strings instead.
        def _fmt_date(d):
            # Portable "Aug 7" style formatting - NOT strftime("%-d"/"%#d"):
            # the no-leading-zero-day flag is spelled differently on Linux/
            # Mac ("%-d") vs Windows ("%#d"), and using the wrong one raises
            # a ValueError at runtime on the other OS. This works identically
            # on every platform since it doesn't rely on either flag.
            return f"{d.strftime('%b')} {d.day}"

        fig.add_vline(
            x=str(SIGNING_DATE), line_dash="dash", line_color="green",
            annotation_text=f"Pact signed ({_fmt_date(SIGNING_DATE)})",
            annotation_position="top left",
        )
        for spike_date in spike_dates:
            fig.add_vline(
                x=str(spike_date), line_dash="dot", line_color="red",
                annotation_text=f"Spike: {_fmt_date(spike_date)}",
                annotation_position="top right",
            )

        fig.update_layout(
            height=480,
            xaxis_title="Date",
            yaxis_title="X posts/day",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, width='stretch')

        if spike_dates:
            spike_list = ", ".join(_fmt_date(d) for d in spike_dates)
            caption = (
                f"{spike_list} {'is' if len(spike_dates) == 1 else 'are'} flagged as volume "
                "spike(s) (posts/day > mean + 1 stdev over the collection window)."
            )
            if latest_data_date is not None and spike_dates[-1] == latest_data_date:
                caption += (
                    f" {_fmt_date(spike_dates[-1])} is also the last day of X collection, so "
                    "it may be a partial day rather than a true peak."
                )
        else:
            caption = "No volume spikes flagged in the current collection window (posts/day > mean + 1 stdev)."
        if n_excluded:
            caption += (
                f" ({n_excluded} stray pre-pact outlier date(s) before {CHART_MIN_DATE} "
                "are excluded from this chart's view - they're still in daily_actor_timeline.csv.)"
            )
        st.caption(caption)
        if gdelt_available:
            gdelt_timeline = load_gdelt_timeline(os.path.getmtime(GDELT_TIMELINE_CSV))
            covered = [a for a in ACTORS if a in gdelt_timeline.columns and gdelt_timeline[a].sum() > 0]
            st.caption(
                f"GDELT coverage in the current dataset: {', '.join(covered) if covered else 'none'}, "
                f"{gdelt_timeline['date'].min()} to {gdelt_timeline['date'].max()}. "
                "Treat GDELT-vs-X comparisons as illustrative only until GDELT coverage matches "
                "the full actor/date range."
            )


# --------------------------------------------------------------------------
# Tab 3: Sentiment & engagement
# --------------------------------------------------------------------------

with tab_sentiment:
    st.subheader("Sentiment vs. engagement, per actor")

    # NOTE: sentiment_score/actors_mentioned always EXIST as columns by the
    # time data reaches here (see OPTIONAL_STAGE_COLUMNS in _load_x_data) -
    # so checking "column not in df" here would never actually catch a
    # not-yet-processed dataset. What matters is whether they have any real
    # values in them, which is what the data-completeness banner above the
    # tabs already checks and explains in detail; this is just the
    # tab-local fallback for the (should be rare) case someone lands here
    # with genuinely zero usable rows.
    if en_df_safe.empty or en_df_safe["sentiment_score"].notna().sum() == 0:
        st.warning(
            "No processed sentiment data available yet for the current filters/dataset - "
            "see the notice above the tabs, or click 🔄 Refresh Data."
        )
    else:
        rows = []
        for actor in ACTORS:
            mask = en_df_safe["actors_list"].apply(lambda lst: actor in lst)
            subset = en_df_safe[mask]
            rows.append({
                "actor": actor,
                "n_posts": len(subset),
                "avg_sentiment": subset["sentiment_score"].mean() if len(subset) else 0,
                "avg_likes": subset["likes"].mean() if len(subset) else 0,
                "avg_retweets": subset["retweets"].mean() if len(subset) else 0,
            })
        report_df = pd.DataFrame(rows)

        col1, col2 = st.columns(2)
        with col1:
            fig_sent = px.bar(
                report_df, x="actor", y="avg_sentiment",
                title="Average sentiment (VADER compound) by actor",
                color="avg_sentiment", color_continuous_scale="RdYlGn",
                range_color=[-0.3, 0.3],
            )
            fig_sent.update_layout(coloraxis_showscale=False, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_sent, width='stretch')
        with col2:
            fig_eng = go.Figure()
            fig_eng.add_trace(go.Bar(x=report_df["actor"], y=report_df["avg_likes"], name="Avg likes"))
            fig_eng.add_trace(go.Bar(x=report_df["actor"], y=report_df["avg_retweets"], name="Avg retweets"))
            fig_eng.update_layout(
                title="Average engagement by actor", barmode="group",
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_eng, width='stretch')

        iran_row = report_df[report_df["actor"] == "Iran"].iloc[0]
        is_lowest_sentiment = report_df["avg_sentiment"].idxmin() == iran_row.name
        is_lowest_engagement = report_df["avg_likes"].idxmin() == iran_row.name
        if is_lowest_sentiment and is_lowest_engagement:
            st.info(
                f"**Iran pattern:** Iran-related posts have the lowest average sentiment "
                f"({iran_row['avg_sentiment']:.2f}, near-neutral) *and* the lowest average "
                f"engagement ({iran_row['avg_likes']:.1f} likes, {iran_row['avg_retweets']:.1f} "
                f"retweets) of all five actors - the other four actors cluster around "
                f"0.20-0.28 average sentiment with meaningfully higher engagement."
            )
        else:
            st.caption(
                "Note: the Iran lowest-sentiment/lowest-engagement pattern reported earlier "
                "no longer holds with the current data - check the chart above for the current numbers."
            )

        st.divider()
        st.caption("At the individual-post level (English subset), by sentiment label:")
        by_label = en_df_safe.groupby("sentiment_label")[["likes", "retweets"]].mean().round(1)
        st.dataframe(by_label, width='stretch')


# --------------------------------------------------------------------------
# Tab 4: Hashtags
# --------------------------------------------------------------------------

with tab_hashtags:
    st.subheader("Top hashtags")

    if not os.path.exists(HASHTAG_FREQ_CSV):
        st.warning(f"{os.path.basename(HASHTAG_FREQ_CSV)} not found.")
    else:
        hashtag_df = load_hashtag_freq(os.path.getmtime(HASHTAG_FREQ_CSV))
        top15 = hashtag_df.sort_values("deduplicated_count", ascending=False).head(15)
        fig = px.bar(
            top15.sort_values("deduplicated_count"),
            x="deduplicated_count", y="hashtag", orientation="h",
            title="Top 15 hashtags by deduplicated count",
        )
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=520)
        st.plotly_chart(fig, width='stretch')
        st.caption(
            "\"Deduplicated count\" treats every post in the same near-duplicate/repost "
            "cluster as one vote, so bot/aggregator clusters don't inflate a hashtag's rank."
        )


# --------------------------------------------------------------------------
# Tab 5: Data explorer
# --------------------------------------------------------------------------

with tab_explorer:
    st.subheader("Raw post explorer")
    st.caption(
        f"{len(en_df_safe)} of {len(df)} total rows shown (English-detected rows, minus "
        f"{(en_df['confirmed_flag'] == True).sum()} rows manually confirmed as slur/"  # noqa: E712
        f"harassment content, which are excluded from display everywhere in this dashboard)."
    )

    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        actor_filter = st.multiselect("Actor mentioned", ACTORS, default=[])
    with filter_col2:
        min_date, max_date = en_df_safe["date"].min(), en_df_safe["date"].max()
        date_range = st.date_input(
            "Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date,
        )
    with filter_col3:
        sentiment_filter = st.multiselect(
            "Sentiment", ["positive", "neutral", "negative"], default=[]
        )

    keyword = st.text_input("Search post text", "")

    filtered = en_df_safe.copy()
    if actor_filter:
        filtered = filtered[filtered["actors_list"].apply(lambda lst: any(a in lst for a in actor_filter))]
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        filtered = filtered[(filtered["date"] >= start_date) & (filtered["date"] <= end_date)]
    if sentiment_filter:
        filtered = filtered[filtered["sentiment_label"].isin(sentiment_filter)]
    if keyword.strip():
        filtered = filtered[filtered["text"].str.contains(keyword, case=False, na=False)]

    st.caption(f"{len(filtered)} matching post(s)")
    display_cols = [
        "timestamp", "username", "text", "likes", "retweets",
        "sentiment_label", "actors_mentioned",
    ]
    st.dataframe(
        filtered[display_cols].sort_values("timestamp", ascending=False),
        width='stretch',
        hide_index=True,
        height=500,
    )


# --------------------------------------------------------------------------
# Refresh polling - MUST be the last thing in the script.
#
# This is deliberately placed here, after every tab above has fully
# rendered, rather than up near the header where the "running" banner is
# shown. st.rerun() halts script execution immediately - calling it right
# after the banner (before the tabs ran) meant the whole rest of the page
# (all 5 tabs, every chart) never rendered at all while a refresh was in
# progress, so there was nothing left to "browse" despite the banner
# saying you could. Putting the sleep+rerun down here instead means the
# full page - header, banner, live log, and all tabs showing the
# last-loaded data - renders completely on every single poll; only once
# that's done do we pause ~2.5s and ask Streamlit to rerun for the next
# poll. When the refresh finishes, the "done" branch above (which runs
# BEFORE data is loaded) already cleared the cache and reloaded fresh
# data, so the very next render after completion shows the newly-fetched
# posts side by side with everything that was already there - not a
# separate "new vs. old" view, just the one updated dataset.
# --------------------------------------------------------------------------

if refresh_status == "running":
    time.sleep(2.5)
    st.rerun()

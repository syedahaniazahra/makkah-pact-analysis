"""
app.py

Phase 8 - Streamlit dashboard for the Makkah/Mecca Joint Defence Pact
project. Reads the already-built analysis outputs (x_data_cleaned.csv,
actor_network_v2.json, actor_centrality.csv, daily_actor_timeline.csv,
hashtag_frequency.csv, daily_gdelt_timeline.csv if present, and
pipeline_status.log for the header timestamp) - this file does no new data
processing, and as of the collection/serving split below, no longer runs
ANY other script either. All the heavy lifting (fetching, cleaning, dedup,
sentiment, relation extraction, network rebuilding) happens entirely in
run_pipeline.py, run independently of this dashboard on its own schedule
(see run_pipeline.py's own docstring, and the Windows Task Scheduler setup
notes delivered alongside it).

--- Collection/serving split ---
This dashboard used to launch its own background subprocess chain
(refresh_worker.py) when someone clicked "Refresh Data" - that button has
been removed entirely, on purpose. This app now ONLY EVER READS existing
files; it never fetches from X, never calls the Groq API, and never
launches any other script. All of that now happens in run_pipeline.py, on
a schedule (e.g. every few hours via Task Scheduler), fully decoupled from
whether this dashboard is even open. The only button left in the header is
"Check for updates", which does nothing more than clear Streamlit's cache
and rerun - no subprocess, no blocking, no fetching - so the page just
re-reads whatever's currently on disk.

Layout: a header (title + a "Data last updated" line sourced from
run_pipeline.py's own pipeline_status.log, plus a "Check for updates"
button) followed by five tabs - Actor Network, Timeline, Sentiment &
Engagement, Hashtags, and a Data Explorer table.

Run:
    streamlit run app.py
"""

import ast
import os
import re
from datetime import datetime

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

X_DATA_CSV = os.path.join(DATA_DIR, "x_data_cleaned.csv")
# Phase 6.5 rebuild: the Actor Network tab reads the typed, weighted
# network (actor_network_v2.json, from build_network_v2.py) plus real
# networkx centrality metrics (actor_centrality.csv), both rebuilt by
# run_pipeline.py on every scheduled cycle.
NETWORK_V2_JSON = os.path.join(DATA_DIR, "actor_network_v2.json")
CENTRALITY_CSV = os.path.join(DATA_DIR, "actor_centrality.csv")
TIMELINE_CSV = os.path.join(DATA_DIR, "daily_actor_timeline.csv")
GDELT_TIMELINE_CSV = os.path.join(DATA_DIR, "daily_gdelt_timeline.csv")
HASHTAG_FREQ_CSV = os.path.join(DATA_DIR, "hashtag_frequency.csv")

# Written by run_pipeline.py (NOT by this dashboard) at the end of every
# scheduled run - see that script's write_status_log(). This is the ONLY
# source used for the "Data last updated" header below; deliberately not
# any single file's mtime, which could be misleading if a step partially
# failed (e.g. clean_x_data.py's output mtime would look "fresh" even on a
# cycle where relation extraction was skipped for a Groq quota wall).
PIPELINE_STATUS_LOG = os.path.join(DATA_DIR, "pipeline_status.log")
# PIPE-separated, not colon-separated: the ISO timestamp field itself
# contains colons (e.g. "2026-09-06T18:01:14Z"), which broke an earlier
# colon-delimited version of this regex - caught by actually rendering the
# dashboard against a sample log, not just eyeballing the format. Must
# match run_pipeline.py's write_status_log() line-for-line.
PIPELINE_RUN_RE = re.compile(r"PIPELINE_RUN\|([^|]+)\|([^|]+)\|(.*)")

ACTORS = ["Iran", "Pakistan", "Saudi Arabia", "United States", "Turkiye"]

# One fixed color per actor, used everywhere an actor needs a consistent
# identity across charts (Timeline lines, summary-strip accents) - a muted,
# desaturated qualitative set chosen deliberately (not a plotting library's
# default categorical palette), so the same actor reads the same color no
# matter which tab or which subset of actors is currently selected.
ACTOR_COLORS = {
    "Iran": "#b25142",
    "Pakistan": "#3f7d5c",
    "Saudi Arabia": "#a9852f",
    "United States": "#3c6e8f",
    "Turkiye": "#7a5a95",
}

# Edge color, by dominant_relation_type, for the Actor Network tab. "unclear"
# has its own muted fallback color (distinct from neutral-reporting's gray)
# in case it's ever the dominant type for a pair - in the current data it
# never is (every one of the 10 pairs' dominant type is either
# neutral-reporting or hostile), but the mapping stays defensive for
# whenever the underlying data changes on a future relation-extraction run.
RELATION_TYPE_COLORS = {
    "hostile": "#b23a3a",           # muted red
    "supportive": "#3f7d5c",        # muted green
    "neutral-reporting": "#8a8a8a",  # gray
    "skeptical": "#c1852f",         # muted amber
    "mediating": "#7a5a95",         # muted plum
    "unclear": "#c9b8a8",           # muted fallback, not used by current data
}
# Display/legend order - "unclear" last since it's the fallback bucket.
RELATION_TYPE_ORDER = ["hostile", "supportive", "skeptical", "neutral-reporting", "mediating", "unclear"]

# Shared chart palette - matches the page's warm-paper background (see
# .streamlit/config.toml) instead of plotly's default stark white, so every
# chart reads as part of the same designed page rather than a plotted-on-
# white-canvas insert. INK is the same near-black-but-warm tone used for
# body text; PLOT_GRID is a soft warm gray, never plotly's default light-blue-gray.
PLOT_BG = "#faf8f5"
PLOT_GRID = "#e3dcce"
INK = "#2b2621"


def _style_fig(fig, legend=True):
    """Applied to every plotly figure right before st.plotly_chart, so chart
    background/gridlines/font stay consistent across all 4 charts instead of
    each one carrying plotly's un-themed defaults (stark white canvas, gray
    sans-serif axis labels) that read as a mismatched insert on the page."""
    fig.update_layout(
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family="IBM Plex Sans, sans-serif", color=INK, size=13),
    )
    # Only touch title_font when a title is actually set - setting it
    # unconditionally on a figure with no title (e.g. the Actor Network
    # graph, which has none) made plotly.js render a literal bold
    # "undefined" string above the chart, caught during the smoke test below.
    if fig.layout.title is not None and fig.layout.title.text:
        fig.update_layout(title_font=dict(family="Source Serif 4, Georgia, serif", size=16, color=INK))
    fig.update_xaxes(gridcolor=PLOT_GRID, zerolinecolor=PLOT_GRID)
    fig.update_yaxes(gridcolor=PLOT_GRID, zerolinecolor=PLOT_GRID)
    return fig
# A couple of stray pre-pact outlier posts (e.g. a 2023 tweet, an old retweet
# surfaced by keyword search) can appear in the raw data with dates far
# before the pact existed. They're kept in the underlying CSVs (no rows are
# dropped from the data itself), but a timeline chart spanning "2023 to now"
# would crush the actual Aug 7-17 window into a sliver - so charts default to
# this floor, with a caption noting anything excluded from the view.
# Shown under every Pearson-r stat on the page, so "what does r mean" is
# answered right where the number is, not only in a hover tooltip - kept to
# one calm line per the "no long notes" rule.
R_EXPLAINER = "r ranges from -1 to +1 - near 0 means little/no linear relationship, near +1 or -1 means a strong one."

CHART_MIN_DATE = pd.Timestamp("2026-08-01").date()
# The pact's actual signing date is a fixed historical fact, not something
# that changes as more data comes in - unlike spike dates (see the Timeline
# tab below), which are now read live from daily_actor_timeline.csv's
# is_spike column instead of being hardcoded here, so they stay correct as
# the collection window grows past whatever day this was last edited.
SIGNING_DATE = pd.Timestamp("2026-08-07").date()

st.set_page_config(page_title="Makkah Pact Monitor", layout="wide")

# --------------------------------------------------------------------------
# Visual system - one deliberate font/color pairing, applied once here,
# rather than Streamlit's stock look. Source Serif 4 for headings (reads
# as editorial/analytical, matches the subject matter) + IBM Plex Sans for
# body/UI text (clean, neutral, highly legible at small chart-label sizes).
# Also hides Streamlit's own chrome (hamburger menu, "Made with Streamlit"
# footer) so the page reads as a finished product, not a generic tool
# shell. No gradients, no card shadows, no uniform rounded-corner grid -
# structure comes from whitespace and thin rules instead.
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', -apple-system, sans-serif;
    }
    h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-family: 'Source Serif 4', Georgia, serif;
        font-weight: 600;
        letter-spacing: -0.01em;
        color: #2b2621;
    }
    h1 {
        font-size: 2.1rem !important;
        padding-bottom: 0.3em;
        border-bottom: 3px solid #1f5f5b;
        display: inline-block;
    }

    #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

    div[data-testid="stMetricValue"] {
        font-family: 'Source Serif 4', Georgia, serif;
        font-weight: 600;
        color: #2b2621;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.8rem;
        color: #6b6b6b;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* Tabs - a visible colored underline on the active tab instead of
       Streamlit's default faint gray indicator, so the current section
       reads clearly as "selected" rather than the whole bar looking flat. */
    .stTabs [data-baseweb="tab"] {
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 500;
        color: #6b6b6b;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #1f5f5b;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #1f5f5b;
        height: 3px;
    }

    /* Buttons - a soft shadow + hover lift so interactive elements read as
       clickable controls, not flat text-on-a-page (paired with the teal
       accent color set in .streamlit/config.toml's primaryColor). */
    .stButton button, .stDownloadButton button {
        box-shadow: 0 1px 3px rgba(43, 38, 33, 0.18);
        transition: box-shadow 0.15s ease, transform 0.15s ease;
    }
    .stButton button:hover, .stDownloadButton button:hover {
        box-shadow: 0 4px 10px rgba(43, 38, 33, 0.22);
        transform: translateY(-1px);
    }

    .finding-callout {
        border-left: 3px solid #1f5f5b;
        padding: 0.7em 1em;
        margin: 0.6em 0;
        background: #f0ece3;
        border-radius: 0 4px 4px 0;
    }
    .finding-callout strong { font-family: 'Source Serif 4', Georgia, serif; }
    </style>
    """,
    unsafe_allow_html=True,
)


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
    "sentiment_score": None, "sentiment_label": "", "topic_relevant": None,
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
def load_network_v2(mtime):
    # encoding="utf-8" is required here, not optional: build_network_v2.py
    # writes this file with encoding="utf-8" and it contains non-ASCII bytes
    # (curly quotes, em-dashes, emoji in the raw post text under
    # mediating_relations). Without an explicit encoding, Python's open()
    # falls back to locale.getpreferredencoding() - on Windows that's the
    # system codepage (cp1252, reported as "charmap"), not UTF-8 - which
    # raises UnicodeDecodeError on those bytes. This is exactly what happened
    # when this was first shipped without the encoding= argument.
    import json
    with open(NETWORK_V2_JSON, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_centrality(mtime):
    return pd.read_csv(CENTRALITY_CSV)


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


@st.cache_data
def load_pipeline_status(mtime):
    """Parse the LAST 'PIPELINE_RUN|<iso>|<status>|<note>' line written by
    run_pipeline.py (see that script's write_status_log()). Returns None if
    the log doesn't exist yet (run_pipeline.py has never run) or has no
    parseable line in it. Only that one line is read for this - not any
    file's mtime - so a partially-failed run is never silently reported as
    if everything succeeded."""
    if not os.path.exists(PIPELINE_STATUS_LOG):
        return None
    with open(PIPELINE_STATUS_LOG, encoding="utf-8") as f:
        text = f.read()
    matches = PIPELINE_RUN_RE.findall(text)
    if not matches:
        return None
    iso_ts, status, note = matches[-1]  # last run in the file wins
    try:
        dt = datetime.strptime(iso_ts, "%Y-%m-%dT%H:%M:%SZ")
        display_ts = dt.strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        display_ts = iso_ts
    return {"timestamp": display_ts, "status": status, "note": note.strip()}


# (refresh_worker.py, refresh.lock, refresh_progress.log, the REFRESH_DONE
# sentinel, and the live-progress polling loop all lived here before this
# rewrite - all of that now lives, in spirit, inside run_pipeline.py, which
# runs completely independently of whether this dashboard is even open.)


# --------------------------------------------------------------------------
# Header: title, "Data last updated" (from run_pipeline.py's status log),
# and a "Check for updates" button that only clears the cache and reruns -
# no subprocess, no fetching, no blocking.
# --------------------------------------------------------------------------

st.title("Makkah / Mecca Joint Defence Pact — Social & News Monitor")

df, _x_mtime = load_x_data()  # mtime only used to bust _load_x_data's cache key; "Data last updated" below comes from pipeline_status.log instead, not this file's mtime
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
    # Kept deliberately short and calm in the UI - the full mechanism (why
    # this happens, exactly which scripts to re-run) lives in
    # CLAUDE_INVESTIGATION_LOG.md for whoever's maintaining the pipeline;
    # a dashboard visitor just needs to know the numbers below are partial
    # right now, not a wall of internal pipeline diagnostics.
    if _incomplete == _en_total:
        st.info("Analysis is still processing for this data - charts will populate on the next scheduled update.")
    elif _incomplete > 0:
        st.caption(f"{_en_total - _incomplete} of {_en_total} posts fully processed; the rest will catch up on the next update.")

pipeline_status = load_pipeline_status(
    os.path.getmtime(PIPELINE_STATUS_LOG) if os.path.exists(PIPELINE_STATUS_LOG) else 0
)

header_left, header_right = st.columns([3, 1])
with header_left:
    if pipeline_status is None:
        st.caption("Data last updated: unknown")
    else:
        status_word = {"success": "", "partial": " (partial update)", "failed": ""}.get(
            pipeline_status["status"], ""
        )
        st.caption(f"Data last updated: {pipeline_status['timestamp']}{status_word}")
with header_right:
    if st.button(
        "Check for updates", width='stretch',
        help="Reloads the current data from disk - collection itself runs on its own schedule.",
    ):
        st.cache_data.clear()
        st.rerun()

st.divider()

# --------------------------------------------------------------------------
# At-a-glance summary strip - the numbers a reader needs before opening any
# tab: how much data, over what window, and which actor stands out on the
# two metrics every other tab elaborates on. Deliberately a plain stat row
# (no boxed cards/shadows) - the whitespace and thin dividers below do the
# separating.
# --------------------------------------------------------------------------
_glance_df = en_df_safe[
    (en_df_safe["topic_relevant"] == True)  # noqa: E712
    & (en_df_safe["date"] >= CHART_MIN_DATE)
]
# Fixed 2026-09-10: this used to run over every topic_relevant==True row with
# no date floor, so a single stray pre-pact outlier (a real Sept 2025 post
# about a related Saudi-Pakistan defense agreement, correctly keyword-matched
# but a year before this pact existed) became the computed "earliest" date -
# producing an impossible-looking "Sep 19 - Sep 9" range once the year was
# dropped from the display. The Timeline tab already excludes exactly this
# class of outlier via CHART_MIN_DATE (see that constant's own comment above)
# - this now applies the same floor here, so the two tell a consistent story
# instead of just the Timeline tab being right.
if len(_glance_df):
    _mention_counts = {a: _glance_df["actors_list"].apply(lambda lst: a in lst).sum() for a in ACTORS}
    _top_actor = max(_mention_counts, key=_mention_counts.get)
    _sent_by_actor = {
        a: _glance_df.loc[_glance_df["actors_list"].apply(lambda lst: a in lst), "sentiment_score"].mean()
        for a in ACTORS
    }
    _sent_by_actor = {a: v for a, v in _sent_by_actor.items() if pd.notna(v)}
    _date_min, _date_max = _glance_df["date"].min(), _glance_df["date"].max()

    # Portable "Aug 7" formatting (not strftime("%-d") - that flag is spelled
    # differently on Windows ("%#d") and raises ValueError on the wrong OS;
    # see the Timeline tab's own _fmt_date for the same fix, applied there).
    # Includes the year only if the range actually spans more than one -
    # defensive against a future collection window crossing a year boundary,
    # even though the CHART_MIN_DATE floor above keeps today's range within one.
    def _fmt_short(d, with_year=False):
        base = f"{d.strftime('%b')} {d.day}"
        return f"{base}, {d.year}" if with_year else base

    _cross_year = _date_min.year != _date_max.year
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Topic-relevant posts", f"{len(_glance_df):,}")
    g2.metric(
        "Collection window",
        f"{_fmt_short(_date_min, _cross_year)} – {_fmt_short(_date_max, _cross_year)}",
    )
    g3.metric("Most-mentioned actor", _top_actor, help=f"{_mention_counts[_top_actor]:,} mentions")
    if _sent_by_actor:
        _most_pos = max(_sent_by_actor, key=_sent_by_actor.get)
        g4.metric("Most positive sentiment", _most_pos, help=f"avg {_sent_by_actor[_most_pos]:.2f}")
    st.divider()

tab_network, tab_timeline, tab_sentiment, tab_hashtags, tab_explorer = st.tabs(
    ["Actor Network", "Timeline", "Sentiment & Engagement", "Hashtags", "Data Explorer"]
)


# --------------------------------------------------------------------------
# Tab 1: Actor network graph
# --------------------------------------------------------------------------

with tab_network:
    st.subheader("Actor relationship network")
    st.caption(
        "Node size = how often that actor appears in an extracted relation. Edge thickness "
        "= relation count for that pair; edge color = the pair's most common relation type "
        "(hover an edge for the full breakdown).",
        help="Relation types are extracted per-post by an LLM classifier, not inferred from "
             "co-occurrence - two actors mentioned in the same post are only linked here if "
             "the post actually describes a relationship between them."
    )

    if not os.path.exists(NETWORK_V2_JSON) or not os.path.exists(CENTRALITY_CSV):
        missing = [os.path.basename(p) for p in (NETWORK_V2_JSON, CENTRALITY_CSV) if not os.path.exists(p)]
        st.warning(
            f"Missing: {', '.join(missing)}. Run build_network_v2.py (needs actor_relations.csv "
            "from a completed relation-extraction pass) to generate them."
        )
    else:
        network_v2 = load_network_v2(os.path.getmtime(NETWORK_V2_JSON))
        centrality_df = load_centrality(os.path.getmtime(CENTRALITY_CSV))
        nodes = network_v2["nodes"]
        edges = network_v2["edges"]

        def _pair_edge(a, b):
            key = tuple(sorted([a, b]))
            for e in edges:
                if tuple(sorted([e["source"], e["target"]])) == key:
                    return e
            return None

        def _format_breakdown(e):
            """'Iran-US: 108 total — 72 hostile, 32 neutral-reporting, 2 skeptical, 2
            mediating, 0 supportive, 0 unclear' - all 6 types listed, highest count
            first, zeros included, so the hover always shows the FULL breakdown."""
            bd = e["relation_type_breakdown"]
            parts = sorted(bd.items(), key=lambda kv: -kv[1])
            breakdown_str = ", ".join(f"{v} {k}" for k, v in parts)
            return f"{e['source']} – {e['target']}: {e['weight']} total — {breakdown_str}"

        G = nx.Graph()
        for n in nodes:
            G.add_node(n["id"], size=n["relation_row_involvement"])
        for e in edges:
            if e["weight"] > 0:
                G.add_edge(e["source"], e["target"], weight=e["weight"])

        pos = nx.spring_layout(G, weight="weight", seed=42, k=1.1)

        max_weight = max((e["weight"] for e in edges), default=1) or 1
        edge_traces = []
        dominant_types_present = set()
        for e in edges:
            if e["weight"] <= 0:
                continue
            x0, y0 = pos[e["source"]]
            x1, y1 = pos[e["target"]]
            width = 1 + 7 * (e["weight"] / max_weight)
            dom = e["dominant_relation_type"]
            color = RELATION_TYPE_COLORS.get(dom, RELATION_TYPE_COLORS["unclear"])
            dominant_types_present.add(dom)
            edge_traces.append(
                go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode="lines",
                    line=dict(width=width, color=color),
                    hoverinfo="text",
                    text=_format_breakdown(e),
                    showlegend=False,
                )
            )

        # Zero-length dummy traces purely to give the color-by-dominant-type
        # mapping a legend entry - only for types actually dominant on at
        # least one of the 10 edges, so the legend doesn't list colors that
        # never appear.
        legend_traces = [
            go.Scatter(
                x=[None], y=[None], mode="lines",
                line=dict(width=4, color=RELATION_TYPE_COLORS.get(rtype, RELATION_TYPE_COLORS["unclear"])),
                name=rtype, showlegend=True, hoverinfo="skip",
            )
            for rtype in RELATION_TYPE_ORDER if rtype in dominant_types_present
        ]

        sizes = [n["relation_row_involvement"] for n in nodes]
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
            hovertext=[f"{n['label']}: {n['relation_row_involvement']} relation-row involvement" for n in nodes],
            hoverinfo="text",
            marker=dict(
                size=[scale_node_size(n["relation_row_involvement"]) for n in nodes],
                color=sizes,
                # Fixed 2026-09-10: plotly's stock "Blues" scale runs light-to-
                # dark, so a lower-involvement actor's node fill lands near-
                # white and all but disappears against a light page background
                # (this is what made Iran/United States hard to see - their
                # involvement values happen to sit at the pale end of the
                # scale). A custom 2-stop scale that never reaches white fixes
                # that regardless of which actor's value is lowest.
                colorscale=[[0, "#a9c9d6"], [1, "#1b4f72"]],
                showscale=False,
                # Each node's OUTLINE is that actor's fixed identity color
                # (same ACTOR_COLORS used on the Timeline chart) - this is
                # what actually guarantees every node reads clearly no matter
                # how light or dark its fill lands on the involvement scale,
                # and it ties this chart into the same color system as the
                # rest of the dashboard instead of using an unrelated palette.
                line=dict(width=3, color=[ACTOR_COLORS.get(n["id"], INK) for n in nodes]),
            ),
            showlegend=False,
        )

        fig = go.Figure(data=edge_traces + legend_traces + [node_trace])
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=540,
            margin=dict(l=10, r=10, t=10, b=40),
            legend=dict(
                orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5,
                title=dict(text="Edge color = dominant relation type: "),
            ),
        )
        st.plotly_chart(_style_fig(fig), width='stretch')

        graph_col, centrality_col = st.columns([3, 2])

        with graph_col:
            with st.expander("Edge breakdown (all 10 pairs)"):
                edge_rows = []
                for e in sorted(edges, key=lambda e: -e["weight"]):
                    row = {
                        "pair": f"{e['source']} – {e['target']}",
                        "weight": e["weight"],
                        "dominant_relation_type": e["dominant_relation_type"],
                    }
                    row.update(e["relation_type_breakdown"])
                    edge_rows.append(row)
                st.dataframe(pd.DataFrame(edge_rows), width='stretch', hide_index=True)

        with centrality_col:
            st.markdown("**Centrality rankings** (degree = weighted node strength)")
            st.dataframe(
                centrality_df.sort_values("degree_centrality", ascending=False),
                width='stretch',
                hide_index=True,
            )
            st.caption(
                "Betweenness is 0.0 for every actor - expected, since all 10 pairs already "
                "have a direct relation, so no third actor sits \"between\" any two others.",
                help="The network is a complete graph: every actor pair has at least one "
                     "extracted relation, so the shortest path between any two actors is "
                     "always that direct edge.",
            )

        # ---- Mediating callout: computed for all 5 actors uniformly, names ----
        # ---- whichever one(s) actually show mediating activity ----
        # Fixed 2026-09-10: this used to hardcode Pakistan-Iran and
        # Pakistan-United States as the only pairs checked, so it always
        # named the same actor regardless of what the data said. Now it sums
        # each actor's mediating-relation involvement across every edge that
        # touches them, and names whichever actor's involvement is highest -
        # currently Pakistan, because that's what the extracted relations
        # actually show, not because the code singles Pakistan out.
        mediating_pairs = [
            (e, e["relation_type_breakdown"].get("mediating", 0))
            for e in edges
            if e["relation_type_breakdown"].get("mediating", 0) > 0
        ]
        if mediating_pairs:
            mediating_by_actor = {a: 0 for a in ACTORS}
            for e, c in mediating_pairs:
                mediating_by_actor[e["source"]] += c
                mediating_by_actor[e["target"]] += c
            top_mediator = max(mediating_by_actor, key=mediating_by_actor.get)
            top_count = mediating_by_actor[top_mediator]
            per_partner = [
                f"{c} with {(e['target'] if e['source'] == top_mediator else e['source'])}"
                for e, c in mediating_pairs if top_mediator in (e["source"], e["target"])
            ]
            plural = "s" if top_count != 1 else ""
            st.markdown(
                f"""<div class="finding-callout">
                <strong>{top_mediator} mediating other actors' relations</strong> ({top_count} instance{plural})<br>
                {', '.join(per_partner)} mediating-typed relation(s), extracted from post text
                describing {top_mediator} as a go-between rather than a direct party.
                </div>""",
                unsafe_allow_html=True,
            )
            total_mediating = sum(c for _, c in mediating_pairs)
            other_mediating = total_mediating - top_count
            if other_mediating:
                st.caption(f"{other_mediating} additional mediating relation(s) elsewhere in the data.")
        else:
            st.caption("No mediating-type relations identified in the current data.")


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
                help="GDELT is a supporting source; coverage may not span all actors or dates.",
            )

        fig = go.Figure()
        for actor in selected_actors:
            if actor in timeline.columns:
                fig.add_trace(go.Scatter(
                    x=timeline["date"], y=timeline[actor],
                    mode="lines+markers", name=actor,
                    line=dict(color=ACTOR_COLORS.get(actor)),
                    marker=dict(color=ACTOR_COLORS.get(actor)),
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
        st.plotly_chart(_style_fig(fig), width='stretch')

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
            caption += f" ({n_excluded} earlier outlier date(s) excluded from view.)"
        st.caption(caption)

        # Correlation between daily X-mention volume and daily GDELT article
        # volume, on dates both sources actually cover - a real Pearson r,
        # not just the visual overlay above. Needs at least a handful of
        # overlapping dates with some variation in both series to mean
        # anything; below that it's silently omitted rather than shown with
        # a caveat paragraph attached.
        #
        # GDELT is a supplementary cross-check, never the analysis itself -
        # the project's required correlation work is entirely X-native (see
        # the Sentiment tab's sentiment-vs-engagement r below, computed only
        # from X data). Labeled "secondary source" here for the same reason
        # the overlay checkbox above already is - and fetch_gdelt.py is a
        # one-time historical backfill (Aug 7-14), not refreshed alongside
        # ongoing X collection, so this r reflects that early window only,
        # not the full collection period.
        if gdelt_available:
            gdelt_timeline = load_gdelt_timeline(os.path.getmtime(GDELT_TIMELINE_CSV))
            merged = timeline_full.merge(gdelt_timeline[["date", "total_articles"]], on="date", how="inner")
            if len(merged) >= 5 and merged["total_posts"].std() > 0 and merged["total_articles"].std() > 0:
                corr = merged["total_posts"].corr(merged["total_articles"])
                st.metric(
                    "X volume vs. GDELT volume correlation (secondary source)", f"r = {corr:.2f}",
                    help=f"Pearson correlation over {len(merged)} overlapping day(s) between daily "
                         "X post volume and daily GDELT article volume. GDELT is a supplementary "
                         "cross-check on a limited early window, not the project's primary analysis.",
                )
                st.caption(R_EXPLAINER)


# --------------------------------------------------------------------------
# Tab 3: Sentiment & engagement
# --------------------------------------------------------------------------

with tab_sentiment:
    st.subheader("Sentiment vs. engagement, per actor")

    # Fixed 2026-09-10: this tab used to run its averages/charts over
    # en_df_safe (every English, non-flagged row), which includes the
    # ~11-12% of rows that are topic_relevant==False - posts that matched
    # a search query but a later Groq pass confirmed aren't actually about
    # the pact. This is what the professor was flagging as sentiment data
    # that looked "not relevant or in line with the Mecca Pact." Scoped
    # down to topic_relevant==True here, matching the same scoping fix
    # applied at the source in add_sentiment.py (see that script's
    # docstring for the full writeup and before/after numbers) - this tab
    # is now reading numbers that were ALSO computed on that same narrower
    # set (add_sentiment.py no longer computes sentiment_score for
    # off-topic rows at all), so this filter is mostly a safety net for
    # older data; the real fix is upstream.
    sentiment_df = en_df_safe[en_df_safe["topic_relevant"] == True]  # noqa: E712 - exact True, excludes False AND NaN
    _n_excluded_sentiment = len(en_df_safe) - len(sentiment_df)
    if _n_excluded_sentiment > 0:
        st.caption(
            f"Scoped to {len(sentiment_df)} topic-relevant English post(s) "
            f"(excludes {_n_excluded_sentiment} English post(s) that matched a "
            "search term but were confirmed off-topic, or not yet classified)."
        )

    # NOTE: sentiment_score/actors_mentioned always EXIST as columns by the
    # time data reaches here (see OPTIONAL_STAGE_COLUMNS in _load_x_data) -
    # so checking "column not in df" here would never actually catch a
    # not-yet-processed dataset. What matters is whether they have any real
    # values in them, which is what the data-completeness banner above the
    # tabs already checks and explains in detail; this is just the
    # tab-local fallback for the (should be rare) case someone lands here
    # with genuinely zero usable rows.
    if sentiment_df.empty or sentiment_df["sentiment_score"].notna().sum() == 0:
        st.caption("No processed sentiment data available yet for this dataset.")
    else:
        rows = []
        for actor in ACTORS:
            mask = sentiment_df["actors_list"].apply(lambda lst: actor in lst)
            subset = sentiment_df[mask]
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
            st.plotly_chart(_style_fig(fig_sent), width='stretch')
        with col2:
            fig_eng = go.Figure()
            fig_eng.add_trace(go.Bar(
                x=report_df["actor"], y=report_df["avg_likes"], name="Avg likes",
                marker_color="#1f5f5b",
            ))
            fig_eng.add_trace(go.Bar(
                x=report_df["actor"], y=report_df["avg_retweets"], name="Avg retweets",
                marker_color="#a9852f",
            ))
            fig_eng.update_layout(
                title="Average engagement by actor", barmode="group",
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(_style_fig(fig_eng), width='stretch')

        # Correlation between sentiment and engagement at the individual-post
        # level (topic-relevant subset) - a real Pearson r alongside the
        # per-actor bar charts above, not just a visual side-by-side. Computed
        # entirely from X data (sentiment_score + likes/retweets, both native
        # to the collected posts) - this is the project's primary correlation
        # result, independent of GDELT.
        _corr_base = sentiment_df.dropna(subset=["sentiment_score"]).copy()
        _corr_base["engagement"] = _corr_base["likes"].fillna(0) + _corr_base["retweets"].fillna(0)
        if len(_corr_base) >= 5 and _corr_base["sentiment_score"].std() > 0 and _corr_base["engagement"].std() > 0:
            sent_eng_corr = _corr_base["sentiment_score"].corr(_corr_base["engagement"])
            st.metric(
                "Sentiment vs. engagement correlation", f"r = {sent_eng_corr:.2f}",
                help=f"Pearson correlation between sentiment_score and (likes + retweets) "
                     f"across {len(_corr_base):,} topic-relevant post(s), all from X data.",
            )
            st.caption(R_EXPLAINER)

        # Fixed 2026-09-10: this used to hardcode "Iran" as the actor to check,
        # so it only ever tested whether Iran specifically was lowest on both
        # measures - if a different actor became lowest on both as new data
        # came in, this callout would have silently stopped appearing instead
        # of naming the actor the data actually points to. Now it finds
        # whichever actor is lowest on each measure, across all five, and
        # only fires (naming that actor) when the same one is lowest on both.
        _low_sentiment_idx = report_df["avg_sentiment"].idxmin()
        _low_engagement_idx = report_df["avg_likes"].idxmin()
        if _low_sentiment_idx == _low_engagement_idx:
            low_row = report_df.loc[_low_sentiment_idx]
            st.markdown(
                f"""<div class="finding-callout">
                <strong>{low_row['actor']} pattern</strong><br>
                {low_row['actor']}-related posts have both the lowest average sentiment
                ({low_row['avg_sentiment']:.2f}) and lowest average engagement
                ({low_row['avg_likes']:.1f} likes, {low_row['avg_retweets']:.1f} retweets) of
                the five actors.
                </div>""",
                unsafe_allow_html=True,
            )

        st.divider()
        st.caption("At the individual-post level (English + topic-relevant subset), by sentiment label:")
        by_label = sentiment_df.groupby("sentiment_label")[["likes", "retweets"]].mean().round(1)
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
        fig.update_traces(marker_color="#1f5f5b")
        st.plotly_chart(_style_fig(fig), width='stretch')
        st.caption(
            "Deduplicated: near-duplicate/repost clusters count once, so bot clusters don't "
            "inflate a hashtag's rank.",
            help="Every post in the same near-duplicate/repost cluster contributes one vote, "
                 "not one per post.",
        )
        # Fixed 2026-09-10: hashtag_frequency.csv/hashtag_cooccurrence.csv are
        # now built by extract_topics_hashtags.py from topic_relevant==True
        # rows only (previously counted every English row) - same fix as
        # Sentiment & Engagement's tab, applied to hashtags. This tab reads
        # the CSV as-is, so surfacing the exclusion count here just makes
        # that upstream scoping visible, not a second filter.
        if "topic_relevant" in en_df_safe.columns:
            _n_hashtag_relevant = int((en_df_safe["topic_relevant"] == True).sum())  # noqa: E712
            _n_hashtag_excluded = len(en_df_safe) - _n_hashtag_relevant
            if _n_hashtag_excluded > 0:
                st.caption(f"Scoped to {_n_hashtag_relevant:,} topic-relevant post(s).")


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

# (This file used to end with a "wait 2.5s then st.rerun()" polling loop
# here, to watch a live-running background refresh. There's no longer
# anything to poll - see the module docstring's "Collection/serving split"
# - so the script just ends after the Data Explorer tab renders.)

"""
refresh_worker.py

Runs the full "Refresh Data" pipeline chain as a plain background process -
this is NOT part of the Streamlit app's own script execution. app.py's
Refresh Data button launches this via subprocess.Popen() (which returns
immediately - it does NOT wait for this to finish) and then polls
refresh_progress.log to show live status, so the dashboard page never
freezes or times out while a refresh runs.

Chain: fetch (--recent) -> [stop early if 0 new rows] -> clean (now
incremental, see clean_x_data.py) -> preprocess -> hashtags -> actor
mentions -> sentiment -> dedupe (always, cheap) -> network rebuild (ONLY
if the deduped actor counts actually changed) -> timeline (--x-only,
GDELT is never touched by a refresh).

Progress is streamed line-by-line as each step's own stdout arrives (not
buffered until a step finishes) - long steps like the X fetch, which has
deliberate multi-second delays between actors/pages to respect rate
limits, print their own "waiting..." lines that show up here live instead
of the log going quiet and looking frozen.

No artificial timeout is applied to any step. The previous version ran
this chain SYNCHRONOUSLY inside a Streamlit button click with a 900s
subprocess timeout, which both froze the page for the whole duration and
could cut off a legitimately slow (rate-limited) X fetch. Since this now
runs as an independent background process instead of inside a web
request, there's no reason to cap it - a genuinely stuck step just means
the log stops advancing, which is visible rather than silently killed.

Ends by writing exactly one sentinel line, which app.py's log-polling
looks for:
    REFRESH_DONE:success:<n> new post(s) added, full pipeline reprocessed in <t>s
    REFRESH_DONE:nochange:no new posts in the lookback window (<t>s)
    REFRESH_DONE:error:<what failed> (after <t>s)

Also creates refresh.lock at start (deletes it when done, success or
failure) - app.py treats "lock file exists" as "a refresh is currently
running", independent of any particular browser tab/session.

Run (normally only ever launched by app.py, not by hand):
    python refresh_worker.py
"""

import os
import re
import subprocess
import sys
import time

import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
REFRESH_LOCK = os.path.join(DATA_DIR, "refresh.lock")
X_CLEANED_CSV = os.path.join(DATA_DIR, "x_data_cleaned.csv")

FETCH_X_SCRIPT = os.path.join(DATA_DIR, "fetch_x_data.py")
CLEAN_SCRIPT = os.path.join(DATA_DIR, "clean_x_data.py")
PREPROCESS_SCRIPT = os.path.join(DATA_DIR, "preprocess_x_text.py")
HASHTAGS_SCRIPT = os.path.join(DATA_DIR, "extract_topics_hashtags.py")
ACTOR_MENTIONS_SCRIPT = os.path.join(DATA_DIR, "extract_actor_mentions.py")
SENTIMENT_SCRIPT = os.path.join(DATA_DIR, "add_sentiment.py")
DEDUPE_SCRIPT = os.path.join(DATA_DIR, "dedupe_actor_cooccurrence.py")
NETWORK_SCRIPT = os.path.join(DATA_DIR, "build_actor_network.py")
TIMELINE_SCRIPT = os.path.join(DATA_DIR, "build_daily_timelines.py")
ACTOR_FREQ_DEDUP_CSV = os.path.join(DATA_DIR, "actor_frequency_deduped.csv")
ACTOR_COOCCUR_DEDUP_CSV = os.path.join(DATA_DIR, "actor_cooccurrence_deduped.csv")

NEW_ROWS_RE = re.compile(r"NEW_ROWS_ADDED:(-?\d+)")


def log(msg):
    print(msg, flush=True)


def run_step(args, label):
    """Run one pipeline script and stream its stdout into our own log AS IT
    ARRIVES (line by line), not all at once after it finishes - this is
    what keeps a slow step (the X fetch especially) from looking frozen.
    `-u` forces the CHILD's own prints to be unbuffered too - by default,
    Python buffers stdout in larger blocks when it's not a terminal (i.e.
    when piped like this), which would otherwise delay even our own
    line-by-line reading here."""
    log(f"\n=== {label} ===")
    proc = subprocess.Popen(
        [sys.executable, "-u", *args],
        cwd=DATA_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_lines = []
    for line in proc.stdout:
        line = line.rstrip("\n")
        output_lines.append(line)
        log(f"    {line}")
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed (exit code {proc.returncode})")
    return "\n".join(output_lines)


def read_text_if_exists(path):
    if os.path.exists(path):
        with open(path) as f:
            return f.read()
    return None


def data_incomplete():
    """True if x_data_cleaned.csv exists but is missing sentiment_score/
    actors_mentioned for ALL of its English rows - i.e. clean_x_data.py was
    run at some point without the rest of the chain (preprocess -> hashtags
    -> actor mentions -> sentiment) following it, which rebuilds
    x_data_cleaned.csv from scratch and clears those later-stage columns.
    Without this check, a Refresh Data click that happens to find 0 new
    posts (--recent only looks back a few days, so this is common) would
    report "nothing to do" and leave the dashboard broken - reprocessing
    needs to happen here even with 0 new posts in that situation."""
    if not os.path.exists(X_CLEANED_CSV):
        return False
    try:
        df = pd.read_csv(X_CLEANED_CSV, dtype={"id": str})
    except Exception:
        return False
    en = df[df.get("detected_language", pd.Series(dtype=str)) == "en"]
    if en.empty:
        return False
    if "sentiment_score" not in df.columns or "actors_mentioned" not in df.columns:
        return True
    return int(en["sentiment_score"].isna().sum()) == len(en)


def main():
    start = time.monotonic()
    with open(REFRESH_LOCK, "w") as f:
        f.write(str(time.time()))

    try:
        fetch_output = run_step(
            [FETCH_X_SCRIPT, "--recent"], "Fetching new posts from X (recent mode)"
        )
        match = NEW_ROWS_RE.search(fetch_output)
        n_new = int(match.group(1)) if match else 0
        log(f"\n>>> {n_new} new post(s) added to x_data.csv")

        if n_new <= 0:
            if not data_incomplete():
                elapsed = time.monotonic() - start
                log(f"REFRESH_DONE:nochange:no new posts in the lookback window ({elapsed:.0f}s)")
                return
            log(
                "\n>>> 0 new posts, but the existing data is missing sentiment/actor-"
                "mention analysis for every English row (clean_x_data.py appears to "
                "have been run without the rest of the chain after it at some point) "
                "- reprocessing the existing data anyway to fix that."
            )

        run_step(
            [CLEAN_SCRIPT],
            "Cleaning data (incremental - only new rows go through dedup clustering/language detection)",
        )
        run_step([PREPROCESS_SCRIPT], "Preprocessing text")
        run_step([HASHTAGS_SCRIPT], "Extracting hashtags & verifying actor matches")
        run_step([ACTOR_MENTIONS_SCRIPT], "Extracting actor mentions")
        run_step([SENTIMENT_SCRIPT], "Running sentiment analysis")

        freq_before = read_text_if_exists(ACTOR_FREQ_DEDUP_CSV)
        cooccur_before = read_text_if_exists(ACTOR_COOCCUR_DEDUP_CSV)
        run_step([DEDUPE_SCRIPT], "Recomputing deduplicated actor counts")
        freq_changed = read_text_if_exists(ACTOR_FREQ_DEDUP_CSV) != freq_before
        cooccur_changed = read_text_if_exists(ACTOR_COOCCUR_DEDUP_CSV) != cooccur_before

        if freq_changed or cooccur_changed:
            run_step([NETWORK_SCRIPT], "Actor counts changed - rebuilding network graph")
        else:
            log("\n(Network graph left as-is - new posts didn't change any actor pair's deduplicated count)")

        run_step([TIMELINE_SCRIPT, "--x-only"], "Rebuilding X timeline (GDELT untouched)")

        elapsed = time.monotonic() - start
        if n_new > 0:
            log(f"REFRESH_DONE:success:{n_new} new post(s) added, full pipeline reprocessed in {elapsed:.0f}s")
        else:
            log(f"REFRESH_DONE:success:no new posts, but existing data was reprocessed to fill in "
                f"missing sentiment/actor analysis, in {elapsed:.0f}s")

    except Exception as exc:
        elapsed = time.monotonic() - start
        log(f"REFRESH_DONE:error:{exc} (after {elapsed:.0f}s)")
    finally:
        try:
            os.remove(REFRESH_LOCK)
        except OSError:
            pass


if __name__ == "__main__":
    main()

"""
test_missing_columns_offline.py

Offline test harness for the fix built after a real scheduled-pipeline
incident: preprocess_x_text.py crashed with an unhandled NLTK LookupError,
and every step downstream of it then crashed too with KeyError:
'clean_text' / 'actors_mentioned' - a cascade of confusing FAILED entries
for one real root cause.

Covers, against real subprocess runs of the actual patched scripts (not
mocks) in an isolated temp working directory:

  1. preprocess_x_text.py: normal case (produces clean_text) - also
     exercises ensure_nltk_resources() for real.
  2. preprocess_x_text.py: 0 English rows - clean success, no crash.
  3. Each downstream script (extract_actor_mentions, add_sentiment,
     build_daily_timelines --x-only, relevance_and_relations): the
     MISSING-required-column case -> exit code 3 + SKIP_REASON, not a
     KeyError. And the NORMAL case (columns present) -> exit 0, still
     works - so this fix doesn't break the working path.
  4. build_network_v2.py: missing actors_mentioned/clean_text -> still
     completes successfully (degrades gracefully, since the core
     network/centrality output doesn't need those columns) vs. normal
     case with them present.
  5. run_pipeline.py's run_step(): the exit-code-3 SKIPPED convention,
     and that a genuine crash's full output (including an asterisk-banner
     style message, matching the real NLTK LookupError shape) is (a)
     never surfaced as a literal row of asterisks in the short summary
     and (b) fully preserved in pipeline_step_errors.log.

Run: python3 test_missing_columns_offline.py
"""

import os
import shutil
import subprocess
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_TO_COPY = [
    "preprocess_x_text.py",
    "extract_actor_mentions.py",
    "add_sentiment.py",
    "build_daily_timelines.py",
    "relevance_and_relations.py",
    "build_network_v2.py",
    "extract_topics_hashtags.py",
    "clean_x_data.py",
    "run_pipeline.py",
    "fetch_x_data.py",
    "setup_x_account.py",
    "fetch_x_test.py",
    "x_auth.py",
]

PASS = []
FAIL = []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  PASS: {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL: {name}  {detail}")


def make_workdir():
    d = tempfile.mkdtemp(prefix="pipeline_fix_test_")
    for fname in SCRIPTS_TO_COPY:
        shutil.copy(os.path.join(SCRIPT_DIR, fname), os.path.join(d, fname))
    return d


def write_csv(path, header, rows):
    import csv as csvmod
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csvmod.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def run(script, workdir, args=None, env_extra=None, timeout=60):
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(
        [sys.executable, script] + (args or []),
        cwd=workdir, capture_output=True, text=True, timeout=timeout, env=env,
    )
    return proc


# ---------------------------------------------------------------------
# 1 & 2: preprocess_x_text.py
# ---------------------------------------------------------------------

def test_preprocess_normal_case():
    print("\n=== preprocess_x_text.py: normal case (produces clean_text) ===")
    d = make_workdir()
    try:
        write_csv(
            os.path.join(d, "x_data_cleaned.csv"),
            ["id", "text", "detected_language", "long_form", "duplicate_content_cluster_id", "confirmed_flag"],
            [
                ["1", "Pakistan and Iran discuss the #MakkahPact RT @someone https://x.co/a", "en", "False", "", ""],
                ["2", "no idea what this is about", "en", "False", "", ""],
                ["3", "publication en francais", "fr", "False", "", ""],
            ],
        )
        proc = run("preprocess_x_text.py", d, timeout=90)
        output = (proc.stdout or "") + (proc.stderr or "")
        print("  exit code:", proc.returncode)
        # This sandbox's own network policy blocks nltk.download() (the
        # same SSRF-protection proxy documented earlier in this project
        # for the device-bridge Linux VM) - so exit 3 with a clean
        # SKIP_REASON here is a REAL, expected outcome in THIS environment
        # specifically, not a bug. What matters is that it's clean (3 +
        # SKIP_REASON), never an unhandled crash (anything else). The
        # user's real Windows machine has normal internet access, so the
        # auto-download itself should actually succeed there - see the
        # separate test_ensure_nltk_resources_unit() below, which proves
        # the download-succeeds branch deterministically without needing
        # live network.
        clean_outcome = proc.returncode == 0 or (proc.returncode == 3 and "SKIP_REASON:" in output)
        check("preprocess_x_text exits cleanly (0, or 3 with SKIP_REASON - never an unhandled crash)",
              clean_outcome, f"exit={proc.returncode}\n{output[-600:]}")
        if proc.returncode == 3:
            print("  (this sandbox has no network for nltk.download() - see note above; treating as expected)")

        import csv as csvmod
        with open(os.path.join(d, "x_data_cleaned.csv"), encoding="utf-8") as f:
            rows = list(csvmod.DictReader(f))
        has_clean_text_col = rows and "clean_text" in rows[0]
        if proc.returncode == 0:
            check("clean_text column written", has_clean_text_col)
            if has_clean_text_col:
                en_row = next(r for r in rows if r["id"] == "1")
                check("clean_text populated for English row",
                      en_row["clean_text"].strip() != "" and "makkahpact" in en_row["clean_text"],
                      f"got: {en_row.get('clean_text')!r}")
                fr_row = next(r for r in rows if r["id"] == "3")
                check("non-English row left blank", fr_row["clean_text"] == "")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_preprocess_zero_english_rows():
    print("\n=== preprocess_x_text.py: 0 English rows ===")
    d = make_workdir()
    try:
        write_csv(
            os.path.join(d, "x_data_cleaned.csv"),
            ["id", "text", "detected_language", "long_form", "duplicate_content_cluster_id", "confirmed_flag"],
            [["1", "texte francais uniquement", "fr", "False", "", ""]],
        )
        proc = run("preprocess_x_text.py", d, timeout=60)
        output = (proc.stdout or "") + (proc.stderr or "")
        # Same sandbox-network caveat as the normal-case test above - a
        # LookupError for a MISSING resource can surface even with 0
        # English rows, since ensure_nltk_resources() checks resources
        # up front, before the row count is even known. Still must never
        # be an unhandled crash.
        clean_outcome = proc.returncode == 0 or (proc.returncode == 3 and "SKIP_REASON:" in output)
        check("exits cleanly with zero English rows (0, or 3 with SKIP_REASON)",
              clean_outcome, f"exit={proc.returncode}\n{output[-600:]}")
        if proc.returncode == 0:
            check("prints the '0 English rows' note", "0 English rows to process" in output, output[-500:])
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_preprocess_unicode_console_safety():
    """Regression test for a REAL scheduled-run incident: preprocess_x_text
    saved clean_text successfully, then crashed with
    UnicodeEncodeError: 'charmap' codec can't encode character '\\U0001f6a8'
    while printing a duplicate-cluster excerpt containing an emoji, because
    Windows' console defaults to a single-byte codepage (cp1252) that can't
    represent it - scraped X/Twitter text is full of emoji, so any print()
    of raw post text was one stray character away from crashing the whole
    step (and being logged as FAILED) even though its real output was fine.
    Fixed via sys.stdout/stderr.reconfigure(encoding="utf-8",
    errors="replace"). Forces the EXACT failure mode via
    PYTHONIOENCODING=cp1252 (this sandbox is Linux, but that env var makes
    Python's stdout encoding behave the same way the user's real Windows
    console default did - confirmed separately, outside this suite, that
    the pre-fix code reproducibly crashes under this exact setup and the
    post-fix code does not)."""
    print("\n=== preprocess_x_text.py: emoji in tweet text doesn't crash the console print (Windows cp1252 repro) ===")
    d = make_workdir()
    try:
        write_csv(
            os.path.join(d, "x_data_cleaned.csv"),
            ["id", "text", "detected_language", "long_form", "duplicate_content_cluster_id", "confirmed_flag",
             "matched_actor", "username"],
            [["1", "BREAKING \U0001F6A8 Pakistan and Saudi Arabia sign pact", "en", "False",
              "clusterA", "False", "Pakistan", "someuser"]],
        )
        proc = run("preprocess_x_text.py", d, timeout=90, env_extra={"PYTHONIOENCODING": "cp1252"})
        output = (proc.stdout or "") + (proc.stderr or "")
        clean_outcome = proc.returncode == 0 or (proc.returncode == 3 and "SKIP_REASON:" in output)
        check("emoji in post text does not crash preprocess_x_text under a cp1252 console",
              clean_outcome, f"exit={proc.returncode}\n{output[-600:]}")
        check("no UnicodeEncodeError in output", "UnicodeEncodeError" not in output, output[-600:])
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_ensure_nltk_resources_unit():
    """Unit-tests preprocess_x_text.ensure_nltk_resources() directly with
    monkeypatched nltk.data.find/nltk.download, so all three branches are
    proven deterministically regardless of whether THIS sandbox happens to
    have network access for a real download (see the network caveat in
    the two tests above - this test does not depend on that)."""
    print("\n=== preprocess_x_text.ensure_nltk_resources(): unit test (mocked) ===")
    d = make_workdir()
    sys.path.insert(0, d)
    try:
        import importlib

        # preprocess_x_text.py calls ensure_nltk_resources() at MODULE
        # IMPORT time (deliberately - see that file's own comment on why:
        # STOPWORDS itself is built at import time too, via NLTK's
        # LazyCorpusLoader). Import it UNMOCKED first - this sandbox
        # actually has real NLTK data now (see the two tests above, which
        # now pass for real) - then only mock nltk.data.find/nltk.download
        # for direct, isolated calls to ensure_nltk_resources() AFTER
        # import. Mocking find() globally before import was tried first
        # and broke: LazyCorpusLoader's OWN internal use of
        # nltk.data.find() for the real stopwords corpus needs a real
        # PathPointer back, not a mocked None - a good reminder that
        # blanket-mocking a shared function used by code outside the
        # thing you're actually testing is asking for a confusing failure.
        import preprocess_x_text as pxt
        importlib.reload(pxt)

        # nltk itself is a shared singleton module (sys.modules["nltk"]) -
        # popping preprocess_x_text from sys.modules and reloading it does
        # NOT reset nltk.data.find/nltk.download, so whatever this test
        # monkeypatches them to would otherwise leak into every later
        # test's import/reload of preprocess_x_text (which calls
        # ensure_nltk_resources() at MODULE level - a stale "always fails"
        # mock left over here would make the next test's plain `import`
        # crash with SystemExit(3) before that test's own code even runs).
        # Save + restore real originals so this test stays self-contained.
        orig_find = pxt.nltk.data.find
        orig_download = pxt.nltk.download

        # Branch 1: everything already present - no download attempted.
        download_calls = []
        pxt.nltk.data.find = lambda path: None  # never raises -> "found"
        pxt.nltk.download = lambda name, quiet=True: download_calls.append(name) or True
        pxt.ensure_nltk_resources()
        check("unit: no download attempted when everything is already present",
              download_calls == [], download_calls)

        # Branch 2: missing, but auto-download succeeds.
        call_count = {"n": 0}
        def find_missing_then_found(path):
            call_count["n"] += 1
            if call_count["n"] <= len(pxt.REQUIRED_NLTK_RESOURCES):
                raise LookupError("not found yet")
            return None
        pxt.nltk.data.find = find_missing_then_found
        download_calls2 = []
        pxt.nltk.download = lambda name, quiet=True: download_calls2.append(name) or True
        pxt.ensure_nltk_resources()
        check("unit: download attempted for each missing resource",
              set(download_calls2) == {name for _, name in pxt.REQUIRED_NLTK_RESOURCES},
              download_calls2)

        # Branch 3: missing, and auto-download fails too -> clean exit(3)
        # with SKIP_REASON, never an unhandled LookupError.
        pxt.nltk.data.find = lambda path: (_ for _ in ()).throw(LookupError("still missing"))
        pxt.nltk.download = lambda name, quiet=True: False
        try:
            pxt.ensure_nltk_resources()
            check("unit: exits when download fails", False, "did not exit at all")
        except SystemExit as e:
            check("unit: exit code 3 when NLTK data unavailable and undownloadable", e.code == 3, e.code)
    finally:
        pxt.nltk.data.find = orig_find
        pxt.nltk.download = orig_download
        sys.path.remove(d)
        sys.modules.pop("preprocess_x_text", None)
        shutil.rmtree(d, ignore_errors=True)


def test_manual_unzip_fallback():
    """Unit-tests _try_manual_unzip() / the updated ensure_nltk_resources()
    fallback branch added after a SECOND real-world incident: the user ran
    `python -m nltk.downloader punkt_tab stopwords wordnet` by hand (which
    reported wordnet as "already up-to-date"), then immediately ran
    run_pipeline.py in the same terminal - and preprocess_x_text STILL
    reported wordnet as unavailable. Reproduced directly in the sandbox: a
    resource's .zip can be valid and present on disk without ever having
    been extracted, and nltk.data.find() does not reliably auto-load
    straight from an unextracted zip. Covers both directions:
      1. zip present but not extracted, download() reports success/no-op
         but find() still fails -> _try_manual_unzip() extracts it and the
         resource becomes findable (no SKIP_REASON, no exit).
      2. nothing available anywhere (no zip, download fails too) -> still
         a clean SKIP_REASON + exit(3), never an unhandled crash."""
    print("\n=== preprocess_x_text._try_manual_unzip(): fallback unit test (mocked) ===")
    d = make_workdir()
    sys.path.insert(0, d)
    try:
        import importlib
        import zipfile

        import preprocess_x_text as pxt
        importlib.reload(pxt)

        # nltk is a shared singleton module - save originals so this
        # test's monkeypatching of find/download/data.path can't leak into
        # whatever test runs next (see the identical note in
        # test_ensure_nltk_resources_unit(), which is where this exact
        # leak was first caught: an unrestored mock there made THIS test's
        # own module reload crash with SystemExit before any of its checks
        # ran).
        orig_find = pxt.nltk.data.find
        orig_download = pxt.nltk.download
        orig_path = list(pxt.nltk.data.path)

        # --- Branch 1: a valid zip sits on disk but was never extracted ---
        fixture_base = os.path.join(d, "nltk_fixture_present")
        zip_dir = os.path.join(fixture_base, "corpora")
        os.makedirs(zip_dir, exist_ok=True)
        inner_dir = os.path.join(d, "_zip_src", "wordnet")
        os.makedirs(inner_dir, exist_ok=True)
        with open(os.path.join(inner_dir, "data.noun"), "w", encoding="utf-8") as f:
            f.write("fake wordnet data for test purposes\n")
        zip_path = os.path.join(zip_dir, "wordnet.zip")
        with zipfile.ZipFile(zip_path, "w") as z:
            z.write(os.path.join(inner_dir, "data.noun"), "wordnet/data.noun")

        extracted_dir = os.path.join(zip_dir, "wordnet")
        check("fixture sanity: zip present, NOT yet extracted",
              os.path.exists(zip_path) and not os.path.isdir(extracted_dir))

        pxt.nltk.data.path = [fixture_base]
        found = pxt._try_manual_unzip("corpora/wordnet")
        check("_try_manual_unzip: reports an extraction was attempted", found is True, found)
        check("_try_manual_unzip: zip contents actually extracted to disk",
              os.path.isfile(os.path.join(extracted_dir, "data.noun")))

        # Full ensure_nltk_resources() path: download() mocked to report
        # success (simulating "already up-to-date") while find() would
        # still raise on the never-extracted zip alone - the fallback must
        # kick in and resolve it without ever hitting SKIP_REASON/exit(3).
        fixture_base2 = os.path.join(d, "nltk_fixture_present2")
        zip_dir2 = os.path.join(fixture_base2, "corpora")
        os.makedirs(zip_dir2, exist_ok=True)
        with zipfile.ZipFile(os.path.join(zip_dir2, "wordnet.zip"), "w") as z:
            z.write(os.path.join(inner_dir, "data.noun"), "wordnet/data.noun")
        pxt.nltk.data.path = [fixture_base2]

        def find_fails_until_extracted(path):
            # Only wordnet is under test here - punkt_tab/stopwords report
            # as already present so the fallback logic under test isn't
            # muddied by them also going missing in this minimal fixture.
            if path != "corpora/wordnet":
                return None
            extracted = os.path.join(zip_dir2, "wordnet")
            if os.path.isdir(extracted):
                return None
            raise LookupError("not found yet")
        pxt.nltk.data.find = find_fails_until_extracted
        pxt.nltk.download = lambda name, quiet=True: True  # "already up-to-date"
        pxt.ensure_nltk_resources()  # must NOT raise SystemExit
        check("ensure_nltk_resources: self-heals via zip fallback when download() "
              "reports success but find() still fails",
              os.path.isdir(os.path.join(zip_dir2, "wordnet")))

        # --- Branch 2: genuinely nothing available anywhere -> still a
        # clean SKIP_REASON + exit(3), the zip fallback must not mask a
        # real "nothing to do" state or hang/crash instead. ---
        empty_base = os.path.join(d, "nltk_fixture_empty")
        os.makedirs(empty_base, exist_ok=True)
        pxt.nltk.data.path = [empty_base]
        pxt.nltk.data.find = lambda path: (_ for _ in ()).throw(LookupError("still missing"))
        pxt.nltk.download = lambda name, quiet=True: False
        try:
            pxt.ensure_nltk_resources()
            check("ensure_nltk_resources: exits when nothing is available anywhere", False,
                  "did not exit at all")
        except SystemExit as e:
            check("ensure_nltk_resources: exit code 3 when zip fallback also finds nothing",
                  e.code == 3, e.code)
    finally:
        pxt.nltk.data.find = orig_find
        pxt.nltk.download = orig_download
        pxt.nltk.data.path = orig_path
        sys.path.remove(d)
        sys.modules.pop("preprocess_x_text", None)
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------
# 3: downstream scripts - missing-column skip vs. normal case
# ---------------------------------------------------------------------

def base_csv_rows(with_clean_text, with_actors_mentioned):
    header = ["id", "text", "detected_language", "timestamp", "likes", "retweets", "matched_actor"]
    if with_clean_text:
        # 2026-09-10: bundled with clean_text on purpose - extract_actor_mentions.py
        # and add_sentiment.py both now also require "topic_relevant" (the
        # topic_relevant-scoping fix), so any fixture asserting their "normal,
        # everything-present" case needs it present too, or those tests would
        # start hitting the missing-column SKIP_REASON path instead of real
        # processing. Set True - this row ("Pakistan mediates between Iran and
        # the United States") is unambiguously on-topic.
        header += ["clean_text", "topic_relevant"]
    if with_actors_mentioned:
        header += ["actors_mentioned"]
    rows = []
    row1 = ["1", "Pakistan mediates between Iran and the United States", "en",
            "2026-09-01T12:00:00Z", "5", "2", "Pakistan"]
    if with_clean_text:
        row1 += ["pakistan mediates between iran and the united states", "True"]
    if with_actors_mentioned:
        row1 += [str(["Iran", "Pakistan", "United States"])]
    rows.append(row1)
    return header, rows


def test_downstream_script(label, script, args, required_cols, extra_header=None, extra_env=None,
                            normal_case_needs_live_network=False):
    print(f"\n=== {script}: missing-column case ({required_cols} absent) ===")
    d = make_workdir()
    try:
        header, rows = base_csv_rows(with_clean_text=False, with_actors_mentioned=False)
        if extra_header:
            header += extra_header[0]
            rows[0] += extra_header[1]
        write_csv(os.path.join(d, "x_data_cleaned.csv"), header, rows)
        proc = run(script, d, args=args, env_extra=extra_env, timeout=60)
        output = (proc.stdout or "") + (proc.stderr or "")
        check(f"{label}: exits 3 when {required_cols} missing", proc.returncode == 3,
              f"exit={proc.returncode}\n{output[-800:]}")
        check(f"{label}: prints SKIP_REASON (not a raw KeyError traceback)",
              "SKIP_REASON:" in output and "KeyError" not in output,
              output[-500:])
    finally:
        shutil.rmtree(d, ignore_errors=True)

    print(f"=== {script}: normal case ({required_cols} present) ===")
    d = make_workdir()
    try:
        header, rows = base_csv_rows(with_clean_text=True, with_actors_mentioned=True)
        if extra_header:
            header += extra_header[0]
            rows[0] += extra_header[1]
        write_csv(os.path.join(d, "x_data_cleaned.csv"), header, rows)
        proc = run(script, d, args=args, env_extra=extra_env, timeout=60)
        output = (proc.stdout or "") + (proc.stderr or "")
        if normal_case_needs_live_network:
            # relevance_and_relations.py calls the real Groq API once past
            # the guard - full completion needs live credentials/network,
            # out of scope for an offline test (this project's established
            # pattern - see test_relevance_and_relations_offline.py). What
            # this test actually needs to prove is that require_columns()
            # does NOT false-positive when the columns ARE present, i.e.
            # it must NOT exit 3 - it should get past the guard and fail
            # later for an unrelated reason (no real API access here).
            check(f"{label}: guard does not false-positive when columns are present (does not exit 3)",
                  proc.returncode != 3, f"exit={proc.returncode}\n{output[-800:]}")
            check(f"{label}: got past the column guard into real processing (no SKIP_REASON)",
                  "SKIP_REASON:" not in output, output[-500:])
        else:
            check(f"{label}: exits 0 in the normal case (no regression)", proc.returncode == 0,
                  f"exit={proc.returncode}\n{output[-800:]}")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_add_sentiment_topic_relevant_scoping():
    """New 2026-09-10: the professor flagged sentiment numbers looking
    "not relevant or in line with the Mecca Pact" - traced to
    add_sentiment.py computing sentiment_score/sentiment_label over EVERY
    detected_language=='en' row, including the ones relevance_and_relations.py
    had already confirmed topic_relevant==False. Fixed to scope on
    detected_language=='en' AND topic_relevant==True. This test builds one
    row of each kind (relevant English, off-topic English, not-yet-classified
    English, non-English) and confirms: sentiment is computed ONLY for the
    relevant row; the other three are left blank (same convention as
    non-English rows always were); and the per-actor report only reflects
    the relevant row."""
    import pandas as pd
    print("\n=== add_sentiment.py: only computes sentiment for topic_relevant==True rows ===")
    d = make_workdir()
    try:
        write_csv(
            os.path.join(d, "x_data_cleaned.csv"),
            ["id", "text", "detected_language", "clean_text", "actors_mentioned",
             "topic_relevant", "likes", "retweets"],
            [
                ["1", "Pakistan and Iran discuss the pact happily", "en",
                 "pakistan and iran discuss the pact happily", str(["Iran", "Pakistan"]), "True", "5", "1"],
                ["2", "Iran wins a football match, unrelated to any pact", "en",
                 "iran wins a football match unrelated to any pact", str(["Iran"]), "False", "3", "0"],
                ["3", "Some other Iran-adjacent post not yet checked", "en",
                 "some other iran adjacent post not yet checked", str(["Iran"]), "", "2", "0"],
                ["4", "Pakistan post in a different language", "fr",
                 "pakistan post in a different language", str(["Pakistan"]), "True", "1", "0"],
            ],
        )
        proc = run("add_sentiment.py", d, timeout=60)
        output = (proc.stdout or "") + (proc.stderr or "")
        check("add_sentiment.py exits 0", proc.returncode == 0, f"exit={proc.returncode}\n{output[-800:]}")

        result = pd.read_csv(os.path.join(d, "x_data_cleaned.csv"), dtype={"id": str})
        result = result.set_index("id")

        check("row 1 (English, topic_relevant=True) got a real sentiment_score",
              pd.notna(result.loc["1", "sentiment_score"]), result.loc["1"].to_dict())
        check("row 1's sentiment_label is set (not blank)",
              isinstance(result.loc["1", "sentiment_label"], str) and result.loc["1", "sentiment_label"] != "",
              result.loc["1", "sentiment_label"])
        check("row 2 (English, topic_relevant=False) is left BLANK - the actual bug being fixed",
              pd.isna(result.loc["2", "sentiment_score"]), result.loc["2"].to_dict())
        check("row 3 (English, topic_relevant not yet classified/NaN) is left blank too",
              pd.isna(result.loc["3", "sentiment_score"]), result.loc["3"].to_dict())
        check("row 4 (non-English, topic_relevant=True) is still left blank - language gate still applies",
              pd.isna(result.loc["4", "sentiment_score"]), result.loc["4"].to_dict())

        check("console output reports the English/relevant/off-topic/unclassified breakdown",
              "1 topic_relevant=True" in output and "1 topic_relevant=False" in output
              and "1 not yet classified" in output, output[:1000])
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_extract_actor_mentions_topic_relevant_scoping():
    """New 2026-09-10, same fix/incident as test_add_sentiment_topic_relevant_scoping()
    above, applied to actor_frequency.csv/actor_cooccurrence.csv: these used
    to count every English row regardless of topic_relevant, diluting actor
    mention/co-occurrence counts with off-topic matches. Fixed to only count
    topic_relevant==True rows in the two aggregate CSVs - while
    actors_mentioned itself (written back to x_data_cleaned.csv) is still
    computed for every English row regardless, since it's a per-row textual
    annotation other parts of the pipeline/dashboard rely on independent of
    relevance filtering."""
    import pandas as pd
    print("\n=== extract_actor_mentions.py: actor_frequency/actor_cooccurrence only count topic_relevant==True ===")
    d = make_workdir()
    try:
        write_csv(
            os.path.join(d, "x_data_cleaned.csv"),
            ["id", "text", "detected_language", "clean_text", "topic_relevant"],
            [
                ["1", "Pakistan and Iran discuss the pact", "en",
                 "pakistan and iran discuss the pact", "True"],
                ["2", "Iran and Turkiye unrelated football result", "en",
                 "iran and turkiye unrelated football result", "False"],
            ],
        )
        proc = run("extract_actor_mentions.py", d, timeout=60)
        output = (proc.stdout or "") + (proc.stderr or "")
        check("extract_actor_mentions.py exits 0", proc.returncode == 0, f"exit={proc.returncode}\n{output[-800:]}")

        result = pd.read_csv(os.path.join(d, "x_data_cleaned.csv"), dtype={"id": str}).set_index("id")
        check("actors_mentioned is still computed for the off-topic row too (per-row field, not scoped)",
              "Iran" in result.loc["2", "actors_mentioned"] and "Turkiye" in result.loc["2", "actors_mentioned"],
              result.loc["2", "actors_mentioned"])

        freq_df = pd.read_csv(os.path.join(d, "actor_frequency.csv")).set_index("actor")
        check("actor_frequency.csv counts Iran once (from the relevant row only, not twice)",
              freq_df.loc["Iran", "count"] == 1, freq_df.loc["Iran", "count"])
        check("actor_frequency.csv does NOT count Turkiye at all (only appears in the off-topic row)",
              freq_df.loc["Turkiye", "count"] == 0, freq_df.loc["Turkiye", "count"])
        check("actor_frequency.csv counts Pakistan once (from the relevant row)",
              freq_df.loc["Pakistan", "count"] == 1, freq_df.loc["Pakistan", "count"])

        cooccur_df = pd.read_csv(os.path.join(d, "actor_cooccurrence.csv"))
        iran_pk = cooccur_df[(cooccur_df["actor_1"] == "Iran") & (cooccur_df["actor_2"] == "Pakistan")]
        check("actor_cooccurrence.csv has Iran-Pakistan=1 (from the relevant row)",
              len(iran_pk) == 1 and iran_pk.iloc[0]["count"] == 1, iran_pk)
        iran_turkiye = cooccur_df[(cooccur_df["actor_1"] == "Iran") & (cooccur_df["actor_2"] == "Turkiye")]
        check("actor_cooccurrence.csv has Iran-Turkiye=0 (the pair only co-occurs in the off-topic row)",
              len(iran_turkiye) == 1 and iran_turkiye.iloc[0]["count"] == 0, iran_turkiye)
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------
# 4: build_network_v2.py - degrade gracefully, don't skip the whole step
# ---------------------------------------------------------------------

def test_build_network_v2():
    print("\n=== build_network_v2.py: actors_mentioned/clean_text missing (degrade, don't crash) ===")
    d = make_workdir()
    try:
        write_csv(
            os.path.join(d, "x_data_cleaned.csv"),
            ["id", "text", "detected_language"],
            [["p1", "some post", "en"]],
        )
        write_csv(
            os.path.join(d, "actor_relations.csv"),
            ["post_id", "actor_1", "actor_2", "relation_type"],
            [
                ["p1", "Iran", "United States", "mediating"],
                ["p1", "Iran", "Pakistan", "hostile"],
                ["p2", "Saudi Arabia", "Turkiye", "supportive"],
                ["p2", "Pakistan", "Saudi Arabia", "neutral-reporting"],
                ["p3", "Iran", "Turkiye", "skeptical"],
                ["p3", "Pakistan", "Turkiye", "unclear"],
                ["p4", "United States", "Saudi Arabia", "hostile"],
                ["p4", "United States", "Turkiye", "supportive"],
                ["p5", "Pakistan", "United States", "mediating"],
                ["p5", "Iran", "Saudi Arabia", "neutral-reporting"],
            ],
        )
        proc = run("build_network_v2.py", d, timeout=60)
        output = (proc.stdout or "") + (proc.stderr or "")
        check("build_network_v2 still exits 0 with columns missing (core output doesn't need them)",
              proc.returncode == 0, f"exit={proc.returncode}\n{output[-800:]}")
        check("prints a note about the missing columns (not silent, not a crash)",
              "Note:" in output and "actors_mentioned" in output, output[-500:])

        import json
        with open(os.path.join(d, "actor_network_v2.json"), encoding="utf-8") as f:
            net = json.load(f)
        check("network JSON still has all 5 nodes / 10 edges despite missing columns",
              len(net["nodes"]) == 5 and len(net["edges"]) == 10,
              f"nodes={len(net['nodes'])} edges={len(net['edges'])}")
        mediating_p1 = next(m for m in net["mediating_relations"] if m["post_id"] == "p1")
        check("mediating entry has null enrichment fields (degraded, not crashed)",
              mediating_p1["actors_mentioned_in_post"] is None and mediating_p1["text_snippet"] is None,
              mediating_p1)
    finally:
        shutil.rmtree(d, ignore_errors=True)

    print("=== build_network_v2.py: normal case (columns present) ===")
    d = make_workdir()
    try:
        write_csv(
            os.path.join(d, "x_data_cleaned.csv"),
            ["id", "text", "detected_language", "actors_mentioned", "clean_text"],
            [["p1", "some post", "en", str(["Iran", "United States"]), "some post about iran and the us"]],
        )
        write_csv(
            os.path.join(d, "actor_relations.csv"),
            ["post_id", "actor_1", "actor_2", "relation_type"],
            [
                ["p1", "Iran", "United States", "mediating"],
                ["p1", "Iran", "Pakistan", "hostile"],
                ["p2", "Saudi Arabia", "Turkiye", "supportive"],
                ["p2", "Pakistan", "Saudi Arabia", "neutral-reporting"],
                ["p3", "Iran", "Turkiye", "skeptical"],
                ["p3", "Pakistan", "Turkiye", "unclear"],
                ["p4", "United States", "Saudi Arabia", "hostile"],
                ["p4", "United States", "Turkiye", "supportive"],
                ["p5", "Pakistan", "United States", "mediating"],
                ["p5", "Iran", "Saudi Arabia", "neutral-reporting"],
            ],
        )
        proc = run("build_network_v2.py", d, timeout=60)
        output = (proc.stdout or "") + (proc.stderr or "")
        check("build_network_v2 exits 0 in the normal case", proc.returncode == 0,
              f"exit={proc.returncode}\n{output[-500:]}")
        import json
        with open(os.path.join(d, "actor_network_v2.json"), encoding="utf-8") as f:
            net = json.load(f)
        mediating_p1 = next(m for m in net["mediating_relations"] if m["post_id"] == "p1")
        check("mediating entry has REAL enrichment fields when columns present",
              mediating_p1["actors_mentioned_in_post"] is not None and mediating_p1["text_snippet"] is not None,
              mediating_p1)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_build_network_v2_unicode_console_safety():
    """Regression test for a SECOND real scheduled-run incident, same root
    cause as preprocess_x_text.py's UnicodeEncodeError but a different
    emoji (a red square, U+1F7E5) and a different print site: this
    script's own console report prints each mediating post's clean_text
    snippet ("text: {m['text_snippet']!r}" in main()), which crashed under
    Windows' cp1252 console default. Forces the exact failure mode via
    PYTHONIOENCODING=cp1252 (confirmed separately, outside this suite,
    that the pre-fix code reproducibly crashes under this exact setup and
    the post-fix code does not)."""
    print("\n=== build_network_v2.py: emoji in mediating post's text_snippet doesn't crash the console print (Windows cp1252 repro) ===")
    d = make_workdir()
    try:
        write_csv(
            os.path.join(d, "x_data_cleaned.csv"),
            ["id", "actors_mentioned", "clean_text"],
            [["p1", str(["Iran", "Pakistan"]), "Iran and Pakistan meet \U0001F7E5 for talks"]],
        )
        write_csv(
            os.path.join(d, "actor_relations.csv"),
            ["post_id", "actor_1", "actor_2", "relation_type"],
            [
                ["p1", "Iran", "Pakistan", "mediating"],
                ["p1", "Iran", "Saudi Arabia", "neutral-reporting"],
                ["p1", "Iran", "Turkiye", "neutral-reporting"],
                ["p1", "Iran", "United States", "neutral-reporting"],
                ["p1", "Pakistan", "Saudi Arabia", "neutral-reporting"],
                ["p1", "Pakistan", "Turkiye", "neutral-reporting"],
                ["p1", "Pakistan", "United States", "neutral-reporting"],
                ["p1", "Saudi Arabia", "Turkiye", "neutral-reporting"],
                ["p1", "Saudi Arabia", "United States", "neutral-reporting"],
                ["p1", "Turkiye", "United States", "neutral-reporting"],
            ],
        )
        proc = run("build_network_v2.py", d, timeout=60, env_extra={"PYTHONIOENCODING": "cp1252"})
        output = (proc.stdout or "") + (proc.stderr or "")
        check("emoji in mediating post text does not crash build_network_v2 under a cp1252 console",
              proc.returncode == 0, f"exit={proc.returncode}\n{output[-600:]}")
        check("no UnicodeEncodeError in output", "UnicodeEncodeError" not in output, output[-600:])
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_extract_topics_hashtags_unicode_console_safety():
    """Regression test for the SAME bug class hit a third time: this
    script's console report prints raw scraped hashtag text
    (freq_df/cooccur_df.to_string() in main()) - and hashtags are
    exactly the kind of user-generated text that can contain an emoji.
    Confirmed separately, outside this suite, that this exact fixture
    (an emoji-containing hashtag arriving via the pre-existing
    "hashtags" column, which parse_existing_hashtags() passes through
    with no character filtering, unlike the regex path) reproducibly
    crashes the pre-fix code under PYTHONIOENCODING=cp1252 and does not
    crash the post-fix code."""
    print("\n=== extract_topics_hashtags.py: emoji in a hashtag doesn't crash the console print (Windows cp1252 repro) ===")
    d = make_workdir()
    try:
        write_csv(
            os.path.join(d, "x_data_cleaned.csv"),
            ["id", "text", "detected_language", "matched_actor", "hashtags", "clean_text",
             "duplicate_content_cluster_id"],
            [["1", "Pakistan and Iran discuss the pact", "en", "Pakistan",
              "#MakkahPact\U0001F7E5, #Iran", "pakistan and iran discuss the pact", ""]],
        )
        proc = run("extract_topics_hashtags.py", d, timeout=60, env_extra={"PYTHONIOENCODING": "cp1252"})
        output = (proc.stdout or "") + (proc.stderr or "")
        check("emoji in a hashtag does not crash extract_topics_hashtags under a cp1252 console",
              proc.returncode == 0, f"exit={proc.returncode}\n{output[-600:]}")
        check("no UnicodeEncodeError in output", "UnicodeEncodeError" not in output, output[-600:])
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_extract_topics_hashtags_topic_relevant_scoping():
    """New 2026-09-10, same fix/incident as
    test_add_sentiment_topic_relevant_scoping() and
    test_extract_actor_mentions_topic_relevant_scoping() above, applied to
    hashtag_frequency.csv/hashtag_cooccurrence.csv: these used to count
    every English row regardless of topic_relevant, diluting hashtag rank
    with off-topic matches - the same "sentiment/hashtag data" complaint the
    professor raised, and the half of it that hadn't been fixed yet. Fixed
    to only count topic_relevant==True rows in the two aggregate CSVs -
    while hashtags_extracted itself (written back to x_data_cleaned.csv) is
    still computed for every English row regardless, same as
    actors_mentioned."""
    import pandas as pd
    print("\n=== extract_topics_hashtags.py: hashtag_frequency/hashtag_cooccurrence only count topic_relevant==True ===")
    d = make_workdir()
    try:
        write_csv(
            os.path.join(d, "x_data_cleaned.csv"),
            ["id", "text", "detected_language", "matched_actor", "hashtags", "clean_text",
             "duplicate_content_cluster_id", "topic_relevant"],
            [
                ["1", "Pakistan and Iran discuss the pact #MakkahPact #Diplomacy", "en", "Pakistan",
                 "", "pakistan and iran discuss the pact", "", "True"],
                ["2", "Iran football result #Football #Iran", "en", "Iran",
                 "", "iran football result", "", "False"],
            ],
        )
        proc = run("extract_topics_hashtags.py", d, timeout=60)
        output = (proc.stdout or "") + (proc.stderr or "")
        check("extract_topics_hashtags.py exits 0", proc.returncode == 0, f"exit={proc.returncode}\n{output[-800:]}")

        result = pd.read_csv(os.path.join(d, "x_data_cleaned.csv"), dtype={"id": str}).set_index("id")
        check("hashtags_extracted is still computed for the off-topic row too (per-row field, not scoped)",
              "#Football" in result.loc["2", "hashtags_extracted"], result.loc["2", "hashtags_extracted"])

        freq_df = pd.read_csv(os.path.join(d, "hashtag_frequency.csv")).set_index("hashtag")
        check("hashtag_frequency.csv counts #MakkahPact (from the relevant row)",
              "#MakkahPact" in freq_df.index and freq_df.loc["#MakkahPact", "raw_count"] == 1, freq_df)
        check("hashtag_frequency.csv does NOT count #Football at all (only appears in the off-topic row)",
              "#Football" not in freq_df.index, freq_df)

        cooccur_df = pd.read_csv(os.path.join(d, "hashtag_cooccurrence.csv"))
        check("hashtag_cooccurrence.csv has the relevant row's pair, not the off-topic row's",
              len(cooccur_df) == 1
              and set(cooccur_df.iloc[0][["hashtag_1", "hashtag_2"]]) == {"#MakkahPact", "#Diplomacy"},
              cooccur_df)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_clean_x_data_normal_case():
    """clean_x_data.py isn't part of the require_columns()/NLTK-fix chain
    tested elsewhere in this file, but it now carries the same UTF-8
    console-safety guard as every other step (see the module docstring
    comment added alongside sys.stdout.reconfigure() there) - this proves
    that addition doesn't break its normal working path. Minimal, valid
    input matching its own required column set."""
    print("\n=== clean_x_data.py: normal case (unaffected by the UTF-8 console-safety addition) ===")
    d = make_workdir()
    try:
        write_csv(
            os.path.join(d, "x_data.csv"),
            ["id", "text", "timestamp", "likes", "retweets", "matched_actor", "username"],
            [["1", "Pakistan and Iran sign the pact \U0001F7E5", "2026-09-01T12:00:00Z",
              "5", "2", "Pakistan", "someuser"]],
        )
        proc = run("clean_x_data.py", d, timeout=60, env_extra={"PYTHONIOENCODING": "cp1252"})
        output = (proc.stdout or "") + (proc.stderr or "")
        check("clean_x_data.py runs cleanly with an emoji in the source text under a cp1252 console",
              proc.returncode == 0, f"exit={proc.returncode}\n{output[-800:]}")
        check("x_data_cleaned.csv was written", os.path.exists(os.path.join(d, "x_data_cleaned.csv")))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_clean_x_data_preserves_relevance_columns():
    """Regression test for the 2026-09-08 incident: clean_x_data.py rebuilds
    x_data_cleaned.csv from x_data.csv from scratch on every run, and its
    historical-row-reuse logic never carried forward relevance_and_relations.py's
    own topic_relevant/relations_checked/relations_recheck_done bookkeeping -
    so every pipeline cycle silently reset it to blank, causing
    relevance_and_relations.py to re-process already-done posts and
    duplicate-append their relations to actor_relations.csv (one real post
    was reprocessed 8 times before this was caught).

    Simulates the real sequence: (1) clean_x_data.py's first run on a fresh
    x_data.csv, producing a row with blank/NaN relevance columns (nothing
    has checked it yet); (2) relevance_and_relations.py "runs" (simulated
    here by directly editing x_data_cleaned.csv, matching what that script
    actually does) and marks the row topic_relevant=True, relations_checked=True;
    (3) clean_x_data.py runs AGAIN on the same unchanged x_data.csv (exactly
    what happens every pipeline cycle) - the row is now historical, and the
    fix must carry its topic_relevant/relations_checked values forward
    instead of wiping them back to blank."""
    print("\n=== clean_x_data.py: preserves topic_relevant/relations_checked for historical rows across a second run ===")
    d = make_workdir()
    try:
        write_csv(
            os.path.join(d, "x_data.csv"),
            ["id", "text", "timestamp", "likes", "retweets", "matched_actor", "username"],
            [["1", "Pakistan and Iran sign the pact", "2026-09-01T12:00:00Z",
              "5", "2", "Pakistan", "someuser"]],
        )

        # First run: nothing has relevance-checked this row yet.
        proc1 = run("clean_x_data.py", d, timeout=60)
        check("first clean_x_data.py run succeeds", proc1.returncode == 0,
              f"exit={proc1.returncode}\n{(proc1.stdout or '') + (proc1.stderr or '')[-800:]}")

        import pandas as pd
        cleaned_path = os.path.join(d, "x_data_cleaned.csv")
        df1 = pd.read_csv(cleaned_path, dtype={"id": str})
        check("topic_relevant column exists after first run", "topic_relevant" in df1.columns, list(df1.columns))
        check("relations_checked column exists after first run", "relations_checked" in df1.columns, list(df1.columns))
        row0_topic_relevant = df1.loc[0, "topic_relevant"] if "topic_relevant" in df1.columns else None
        check("row is blank/NaN for topic_relevant before relevance_and_relations.py has run",
              pd.isna(row0_topic_relevant), row0_topic_relevant)

        # Simulate relevance_and_relations.py having processed this row -
        # exactly the columns/values it would set. Cast to object dtype
        # first: pandas infers float64 for an all-NaN column on read, which
        # rejects an assigned True (this is purely a test-fixture wrinkle,
        # not something the real relevance_and_relations.py hits, since it
        # builds these columns itself rather than reading them all-NaN).
        df1["topic_relevant"] = df1["topic_relevant"].astype(object)
        df1["relations_checked"] = df1["relations_checked"].astype(object)
        df1.loc[0, "topic_relevant"] = True
        df1.loc[0, "relations_checked"] = True
        df1.to_csv(cleaned_path, index=False)

        # Second run on the SAME unchanged x_data.csv - this is what happens
        # every pipeline cycle. The row is now historical (already in
        # load_previous_cleaned()'s prev_lookup).
        proc2 = run("clean_x_data.py", d, timeout=60)
        check("second clean_x_data.py run succeeds", proc2.returncode == 0,
              f"exit={proc2.returncode}\n{(proc2.stdout or '') + (proc2.stderr or '')[-800:]}")

        df2 = pd.read_csv(cleaned_path, dtype={"id": str})
        check("topic_relevant survives a second clean_x_data.py run (THE bug this fix addresses)",
              bool(df2.loc[0, "topic_relevant"]) is True, df2.loc[0, "topic_relevant"])
        check("relations_checked survives a second clean_x_data.py run",
              bool(df2.loc[0, "relations_checked"]) is True, df2.loc[0, "relations_checked"])
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_append_relations_dedup_safeguard():
    """Unit test for the 2026-09-08 hard safeguard added to
    relevance_and_relations.py's append_relations(): even if a bug
    elsewhere causes the same post to be resubmitted, the EXACT same
    (post_id, actor_1, actor_2) combination must never be written to
    actor_relations.csv twice. Directly imports the module and calls
    append_relations() - no Groq API calls involved."""
    print("\n=== relevance_and_relations.py: append_relations() dedup safeguard ===")
    d = make_workdir()
    sys.path.insert(0, d)
    try:
        import importlib
        # relevance_and_relations.py instantiates a Groq client only at
        # actual call time inside main(), not at import time, so importing
        # it with no GROQ_API_KEY set is safe (matches the existing
        # test_downstream_script coverage for this same script elsewhere
        # in this file, which also imports/runs it without a real key).
        import relevance_and_relations as rar
        importlib.reload(rar)
        rar.DATA_DIR = d
        rar.RELATIONS_CSV = os.path.join(d, "actor_relations.csv")

        # First append: 2 genuinely new rows -> both written, header created.
        rar.append_relations([
            ("post1", "Iran", "Pakistan", "neutral-reporting"),
            ("post2", "Saudi Arabia", "Turkiye", "cooperative"),
        ])
        df = __import__("pandas").read_csv(rar.RELATIONS_CSV)
        check("first append_relations() call writes both new rows", len(df) == 2, df)

        # Second append: post1/Iran/Pakistan is an EXACT repeat (simulates
        # the real incident - the same post reprocessed after clean_x_data.py
        # wiped relations_checked), post3 is genuinely new. Also includes an
        # exact duplicate of post3 WITHIN this same call, to check the
        # within-call dedup path too.
        rar.append_relations([
            ("post1", "Iran", "Pakistan", "hostile"),  # exact combo repeat, different relation_type
            ("post3", "United States", "Iran", "hostile"),
            ("post3", "United States", "Iran", "hostile"),  # duplicate within this same call
        ])
        df2 = __import__("pandas").read_csv(rar.RELATIONS_CSV)
        check("append_relations() skips an exact (post_id, actor_1, actor_2) repeat "
              "instead of duplicating it", len(df2) == 3, df2)
        check("the original post1/Iran/Pakistan row is untouched (not overwritten with "
              "the later contradictory relation_type)",
              df2[(df2["post_id"] == "post1")]["relation_type"].iloc[0] == "neutral-reporting", df2)
        check("post3/United States/Iran was written exactly once despite appearing twice "
              "in the same call", len(df2[df2["post_id"] == "post3"]) == 1, df2)
        check("actor_relations.csv still has no duplicate (post_id, actor_1, actor_2) rows",
              not df2.duplicated(subset=["post_id", "actor_1", "actor_2"]).any(), df2)
    finally:
        sys.path.remove(d)
        sys.modules.pop("relevance_and_relations", None)
        shutil.rmtree(d, ignore_errors=True)


def test_fetch_x_data_diagnostics_log():
    """Unit test for the diagnostic-logging addition to fetch_x_data.py:
    fetch_actor() returns (tweets, error) instead of just tweets, and
    append_diagnostics() writes one durable block per run to
    fetch_diagnostics.log - covering BOTH the all-actors-ok case and the
    case where an actor's search raises (simulating a degraded/expired
    session, an account lock, etc.).

    2026-09-10: updated for the twscrape -> twifork/twikit pivot (see
    x_auth.py's docstring and CLAUDE_INVESTIGATION_LOG.md) - fetch_actor()
    now takes a twikit-style client whose search_tweet() returns a paged
    Result object with an async .next(), not an async generator, so the
    fake client below mimics THAT shape instead. No real network calls."""
    print("\n=== fetch_x_data.py: diagnostic logging (fetch_actor return shape + append_diagnostics) ===")
    d = make_workdir()
    sys.path.insert(0, d)
    try:
        import asyncio
        import importlib
        import fetch_x_data as fxd
        importlib.reload(fxd)
        fxd.DIAG_LOG = os.path.join(d, "fetch_diagnostics.log")
        fxd.INTRA_SEARCH_DELAY_MIN_SEC = 0
        fxd.INTRA_SEARCH_DELAY_MAX_SEC = 0.01

        class FakeUser:
            screen_name = "someuser"

        class FakeTweet:
            def __init__(self, tid):
                self.id = tid
                self.full_text = f"post {tid}"
                self.user = FakeUser()
                self.created_at_datetime = "2026-09-08T00:00:00Z"
                self.favorite_count = 1
                self.retweet_count = 0
                self.hashtags = []

        class FakePage:
            """Mimics twikit's Result: iterable, __len__, async .next()."""
            def __init__(self, items):
                self._items = items
            def __iter__(self):
                return iter(self._items)
            def __len__(self):
                return len(self._items)
            async def next(self):
                return FakePage([])  # one page only, then exhausted

        class FakeClientOk:
            async def search_tweet(self, query, product, count=20):
                return FakePage([FakeTweet(f"id{i}") for i in range(3)])

        class FakeClientBroken:
            async def search_tweet(self, query, product, count=20):
                # Simulates a degraded/expired session raising - e.g. the
                # real XClIdAccountError/XClIdParseError-class failure this
                # whole pivot was about - before returning anything.
                raise RuntimeError("simulated: account unauthorized (session expired)")

        # --- fetch_actor(): success path ---
        tweets, error = asyncio.run(fxd.fetch_actor(FakeClientOk(), "Iran", ["iran"]))
        check("fetch_actor() returns 3 tweets on the success path", len(tweets) == 3, tweets)
        check("fetch_actor() returns error=None on the success path", error is None, error)

        # --- fetch_actor(): the actual failure path this was added for ---
        tweets2, error2 = asyncio.run(fxd.fetch_actor(FakeClientBroken(), "Pakistan", ["pakistan"]))
        check("fetch_actor() returns 0 tweets when the search raises", tweets2 == [], tweets2)
        check("fetch_actor() surfaces the exception text instead of swallowing it silently",
              error2 is not None and "unauthorized" in error2, error2)

        # --- append_diagnostics(): both an all-ok run and a run with an error ---
        fxd.append_diagnostics(
            mode_desc="recent", since_date="2026-09-05", limit_per_actor=40,
            actor_results={"Iran": {"count": 3, "error": None}},
            total_before=10, total_after=13,
        )
        fxd.append_diagnostics(
            mode_desc="recent", since_date="2026-09-05", limit_per_actor=40,
            actor_results={"Pakistan": {"count": 0, "error": "simulated: account unauthorized (session expired)"}},
            total_before=13, total_after=13,
        )
        check("fetch_diagnostics.log was created", os.path.exists(fxd.DIAG_LOG))
        with open(fxd.DIAG_LOG, encoding="utf-8") as f:
            log_text = f.read()
        check("log records the ok run's tweet count", "Iran: 3 tweet(s) returned [ok]" in log_text, log_text)
        check("log records the errored run's actor and exception text",
              "Pakistan: 0 tweet(s) returned [ERROR: simulated: account unauthorized" in log_text, log_text)
        check("log flags a run where an actor errored, even though it still 'succeeded' overall",
              "ONE OR MORE ACTORS ERRORED" in log_text, log_text)
    finally:
        sys.path.remove(d)
        sys.modules.pop("fetch_x_data", None)
        shutil.rmtree(d, ignore_errors=True)


def test_fetch_actor_pagination():
    """New 2026-09-10, added with the twifork/twikit pivot: fetch_actor()
    now paginates through real page boundaries via a Result's async
    .next() (twikit hands back ~20 tweets per underlying request), instead
    of twscrape's approximate "pace every 20 yielded" scheme. Confirms:
    (1) it stops exactly at LIMIT_PER_ACTOR even mid-page, without
    overshooting into the next page's tweets; (2) it stops cleanly when a
    page comes back empty, even if LIMIT_PER_ACTOR was never reached
    (this is why the loop checks `while page:`, i.e. len(page) != 0, and
    not the page's cursor - twikit's own Result docstring warns that
    looping on the cursor instead can spin forever, since X hands back a
    valid-looking cursor even on an empty page)."""
    print("\n=== fetch_x_data.py: fetch_actor() pagination (limit cutoff + empty-page stop) ===")
    d = make_workdir()
    sys.path.insert(0, d)
    try:
        import asyncio
        import importlib
        import fetch_x_data as fxd
        importlib.reload(fxd)
        fxd.INTRA_SEARCH_DELAY_MIN_SEC = 0
        fxd.INTRA_SEARCH_DELAY_MAX_SEC = 0.01

        class FakeUser:
            screen_name = "someuser"

        class FakeTweet:
            _counter = 0
            def __init__(self):
                FakeTweet._counter += 1
                self.id = FakeTweet._counter
                self.full_text = f"tweet {self.id}"
                self.user = FakeUser()
                self.created_at_datetime = "2026-09-10T00:00:00Z"
                self.favorite_count = 1
                self.retweet_count = 2
                self.hashtags = []

        def make_client(page_size, max_pages):
            """max_pages counts calls to .next(); the (max_pages)-th call
            (0-indexed) returns an empty page - simulates real exhaustion."""
            calls = [0]

            class FakePage:
                def __init__(self, items):
                    self._items = items
                def __iter__(self):
                    return iter(self._items)
                def __len__(self):
                    return len(self._items)
                async def next(self):
                    calls[0] += 1
                    if calls[0] >= max_pages:
                        return FakePage([])
                    return FakePage([FakeTweet() for _ in range(page_size)])

            class FakeClient:
                async def search_tweet(self, query, product, count=20):
                    return FakePage([FakeTweet() for _ in range(page_size)])

            return FakeClient()

        # --- cutoff mid-page: limit (8) falls inside what would be the 2nd page (5+5) ---
        FakeTweet._counter = 0
        fxd.LIMIT_PER_ACTOR = 8
        tweets, error = asyncio.run(fxd.fetch_actor(make_client(page_size=5, max_pages=10), "A", ["a"]))
        check("stops exactly at LIMIT_PER_ACTOR even mid-page (no overshoot)",
              error is None and len(tweets) == 8, (error, len(tweets)))

        # --- exhausts all real pages before reaching an unreachably high limit ---
        FakeTweet._counter = 0
        fxd.LIMIT_PER_ACTOR = 1000
        tweets2, error2 = asyncio.run(fxd.fetch_actor(make_client(page_size=5, max_pages=3), "B", ["b"]))
        check("stops cleanly on an empty page when the limit is never reached "
              "(3 real pages of 5 = 15, not stuck looping)",
              error2 is None and len(tweets2) == 15, (error2, len(tweets2)))
    finally:
        sys.path.remove(d)
        sys.modules.pop("fetch_x_data", None)
        shutil.rmtree(d, ignore_errors=True)


def test_x_auth_cookie_loading():
    """New 2026-09-10, added with the twscrape -> twifork/twikit pivot
    (see x_auth.py's docstring and CLAUDE_INVESTIGATION_LOG.md). Replaces
    the old accounts.db-focused setup_x_account.py tests - there's no
    local account pool/database anymore, so what actually needs covering
    is x_auth.py's .env-reading logic itself: (1) the X_AUTH_TOKEN/X_CT0
    fallback builds the right {auth_token, ct0} dict; (2) X_COOKIES, when
    set, takes priority over the fallback vars and every cookie in the
    string survives into the dict (not just auth_token/ct0); (3) missing
    everything fails loudly (SystemExit with a clear message) instead of
    silently building an empty/broken client. Pure in-process import, no
    subprocess and no network."""
    print("\n=== x_auth.py: .env cookie loading (fallback, X_COOKIES priority, missing-creds error) ===")
    d = make_workdir()
    sys.path.insert(0, d)
    try:
        import importlib

        # --- fallback path: only X_AUTH_TOKEN/X_CT0 set ---
        os.environ.pop("X_COOKIES", None)
        os.environ["X_AUTH_TOKEN"] = "fallbackAAAAAAAAAAAAAAAAAAAAAAAAAAA1"
        os.environ["X_CT0"] = "fallbackBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
        import x_auth
        importlib.reload(x_auth)
        cookies = x_auth.load_cookie_dict()
        check("fallback (X_AUTH_TOKEN/X_CT0) builds the right {auth_token, ct0} dict",
              cookies == {"auth_token": os.environ["X_AUTH_TOKEN"], "ct0": os.environ["X_CT0"]}, cookies)

        # --- X_COOKIES takes priority, extra cookies survive intact ---
        full_cookie_string = (
            "guest_id=v1%3A999888777; "
            "auth_token=fullstringEEEEEEEEEEEEEEEEEEEEEEEEEEEE3; "
            "ct0=fullstringFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF; "
            "twid=u%3D1234567890; "
            "personalization_id=\"v1_abcdefghijklmnop==\""
        )
        os.environ["X_COOKIES"] = full_cookie_string
        cookies2 = x_auth.load_cookie_dict()
        check("X_COOKIES takes priority over the (different) X_AUTH_TOKEN/X_CT0 fallback values",
              cookies2.get("auth_token", "").startswith("fullstring")
              and cookies2.get("ct0", "").startswith("fullstring"), cookies2)
        check("guest_id (a cookie the fallback mode never sends) survived into the dict",
              "guest_id" in cookies2, cookies2)
        check("twid survived into the dict", "twid" in cookies2, cookies2)
        check("personalization_id survived into the dict", "personalization_id" in cookies2, cookies2)

        # --- missing everything: fails loudly, not silently ---
        os.environ.pop("X_COOKIES", None)
        os.environ.pop("X_AUTH_TOKEN", None)
        os.environ.pop("X_CT0", None)
        raised = False
        try:
            x_auth.load_cookie_dict()
        except SystemExit:
            raised = True
        check("missing X_COOKIES and X_AUTH_TOKEN/X_CT0 raises SystemExit instead of "
              "silently building a broken client", raised)
    finally:
        for var in ("X_COOKIES", "X_AUTH_TOKEN", "X_CT0"):
            os.environ.pop(var, None)
        sys.path.remove(d)
        sys.modules.pop("x_auth", None)
        shutil.rmtree(d, ignore_errors=True)


def test_x_auth_build_client_loads_cookies_onto_real_client():
    """New 2026-09-10: confirms build_client() end-to-end against the REAL
    installed twikit (via the twifork package) - not just the dict-parsing
    logic above - by constructing a real Client, calling set_cookies() on
    it, then reading them straight back via the client's own
    get_cookies(). No network call is made by set_cookies()/get_cookies(),
    so this is still a safe, offline check, but it does prove the
    twifork[impersonate] dependency (curl_cffi Chrome impersonation) is
    actually installed and importable - the exact thing that would be
    silently broken if requirements.txt drifted again the way twscrape's
    unpinned version did."""
    print("\n=== x_auth.py: build_client() actually loads .env cookies onto a real twikit Client ===")
    d = make_workdir()
    sys.path.insert(0, d)
    try:
        import importlib
        os.environ.pop("X_AUTH_TOKEN", None)
        os.environ.pop("X_CT0", None)
        os.environ["X_COOKIES"] = "auth_token=zzz999; ct0=yyy888; guest_id=v1%3Afoo"
        import x_auth
        importlib.reload(x_auth)
        client = x_auth.build_client()
        got = client.get_cookies()
        check("build_client() actually sets the .env cookies onto the real Client object",
              got.get("auth_token") == "zzz999" and got.get("ct0") == "yyy888"
              and got.get("guest_id") == "v1%3Afoo", got)
        check("build_client() exposes search_tweet (what fetch_x_data.py/fetch_x_test.py call)",
              hasattr(client, "search_tweet"))
        check("build_client() exposes is_logged_in (what setup_x_account.py calls)",
              hasattr(client, "is_logged_in"))
    finally:
        os.environ.pop("X_COOKIES", None)
        sys.path.remove(d)
        sys.modules.pop("x_auth", None)
        shutil.rmtree(d, ignore_errors=True)


def test_x_auth_cookie_jar_persistence():
    """New 2026-09-10 (later same day), added after a real Task Scheduler
    failure traced to fetch_x_data.py always re-reading a static .env
    cookie snapshot, even though X had likely rotated ct0 since that
    snapshot was hand-pasted (see CLAUDE_INVESTIGATION_LOG.md for the full
    incident writeup). Covers save_session_cookies()/build_client()'s new
    jar-preferring behavior:
      1. save_session_cookies() writes a jar + a meta file tagging it with
         a fingerprint of the .env cookies used.
      2. A second build_client() call, same .env, loads the JAR's cookies
         (not .env's) - proven by mutating the client's cookies via
         set_cookies() before saving, so jar content provably differs
         from raw .env content once loaded back.
      3. If .env cookies then change (simulating a fresh manual paste),
         build_client() detects the fingerprint mismatch and falls back to
         the NEW .env cookies, ignoring the now-stale jar - a fresh manual
         refresh must always take over immediately, never get stuck behind
         old persisted state."""
    print("\n=== x_auth.py: cookie jar persistence (save_session_cookies() / build_client() jar preference) ===")
    d = make_workdir()
    sys.path.insert(0, d)
    orig_cwd = os.getcwd()
    try:
        # COOKIE_JAR_PATH/COOKIE_JAR_META_PATH are relative paths (same
        # convention as every other *_CSV constant in this codebase, which
        # normally lands in the right place because subprocess runs use
        # cwd=d - see run()). This test calls x_auth's functions in-process
        # instead, so it has to chdir here itself to get the same effect.
        os.chdir(d)
        import importlib
        os.environ.pop("X_AUTH_TOKEN", None)
        os.environ.pop("X_CT0", None)
        os.environ["X_COOKIES"] = "auth_token=orig111; ct0=orig222"
        import x_auth
        importlib.reload(x_auth)

        # --- 1: save_session_cookies() writes jar + meta ---
        client1 = x_auth.build_client()
        # Simulate X having rotated ct0 mid-session by setting a DIFFERENT
        # cookie value on the live client before saving - this is what
        # save_session_cookies() should capture, not the original .env value.
        client1.set_cookies({"auth_token": "orig111", "ct0": "ROTATED333"})
        x_auth.save_session_cookies(client1)
        check("save_session_cookies() wrote the cookie jar file",
              os.path.exists(os.path.join(d, x_auth.COOKIE_JAR_PATH)))
        check("save_session_cookies() wrote the fingerprint meta file",
              os.path.exists(os.path.join(d, x_auth.COOKIE_JAR_META_PATH)))

        # --- 2: same .env -> next build_client() loads the JAR (rotated ct0), not raw .env ---
        client2 = x_auth.build_client()
        got2 = client2.get_cookies()
        check("build_client() loaded the persisted (rotated) ct0 from the jar, not .env's original value",
              got2.get("ct0") == "ROTATED333", got2)

        # --- 3: .env cookies change -> stale jar is detected and ignored ---
        os.environ["X_COOKIES"] = "auth_token=freshFFFF; ct0=freshGGGG"
        client3 = x_auth.build_client()
        got3 = client3.get_cookies()
        check("a fresh .env paste takes over immediately, ignoring the now-stale jar",
              got3.get("auth_token") == "freshFFFF" and got3.get("ct0") == "freshGGGG", got3)
    finally:
        os.chdir(orig_cwd)
        os.environ.pop("X_COOKIES", None)
        sys.path.remove(d)
        sys.modules.pop("x_auth", None)
        shutil.rmtree(d, ignore_errors=True)


def test_x_auth_is_logged_in_with_retry():
    """New 2026-09-10 (later same day): is_logged_in()==False isn't proof
    of a dead session - it's the same signal twifork uses for the known,
    transient Cloudflare-challenge-shell blip (see CLAUDE_INVESTIGATION_LOG.md).
    Covers is_logged_in_with_retry()'s three real shapes: succeeds
    immediately (no wasted retry), fails then succeeds on retry (the
    transient-blip case this was built for), and fails every attempt (a
    genuinely dead session, reported only after retries are exhausted).
    Uses a tiny stub client (not a real twikit Client) since
    is_logged_in_with_retry() only ever calls client.is_logged_in() - and
    delay_sec=0.01 so this doesn't actually pause the test suite."""
    import asyncio
    print("\n=== x_auth.py: is_logged_in_with_retry() (transient-blip vs genuinely-dead session) ===")
    d = make_workdir()
    sys.path.insert(0, d)
    try:
        import importlib
        import x_auth
        importlib.reload(x_auth)

        class StubClient:
            def __init__(self, results):
                self._results = list(results)
                self.calls = 0

            async def is_logged_in(self):
                self.calls += 1
                return self._results.pop(0)

        # --- succeeds immediately: no retry spent ---
        c1 = StubClient([True, True])
        ok1 = asyncio.run(x_auth.is_logged_in_with_retry(c1, retries=1, delay_sec=0.01))
        check("returns True immediately when the first check succeeds", ok1 is True)
        check("does not waste a retry when the first check already succeeded", c1.calls == 1, c1.calls)

        # --- fails once, succeeds on retry: the transient-blip case ---
        c2 = StubClient([False, True])
        ok2 = asyncio.run(x_auth.is_logged_in_with_retry(c2, retries=1, delay_sec=0.01))
        check("returns True after a transient False followed by a True on retry", ok2 is True)
        check("made exactly 2 calls (initial + 1 retry)", c2.calls == 2, c2.calls)

        # --- fails every attempt: genuinely dead, reported only after retries exhausted ---
        c3 = StubClient([False, False])
        ok3 = asyncio.run(x_auth.is_logged_in_with_retry(c3, retries=1, delay_sec=0.01))
        check("returns False only once every attempt (initial + all retries) has failed", ok3 is False)
        check("made exactly 2 calls (initial + 1 retry) before giving up", c3.calls == 2, c3.calls)
    finally:
        sys.path.remove(d)
        sys.modules.pop("x_auth", None)
        shutil.rmtree(d, ignore_errors=True)


def test_setup_x_account_missing_credentials_exits_cleanly():
    """New 2026-09-10, replaces the old accounts.db-refresh subprocess
    tests (there's no local account pool to refresh anymore - see
    x_auth.py's docstring). What's still worth a real subprocess check:
    setup_x_account.py exits non-zero with a clear message when .env has
    neither X_COOKIES nor X_AUTH_TOKEN/X_CT0, WITHOUT ever attempting a
    network call (this is exercised before build_client() constructs a
    Client, let alone calls is_logged_in()). Verifying an actual
    successful login is inherently a live-network integration check now
    (is_logged_in() makes a real request to X by design) - not something
    this offline suite can cover; that's confirmed by the user running the
    script for real, per this project's standing rule of always saying
    whether a change was run here or needs a manual run."""
    print("\n=== setup_x_account.py: missing .env credentials exits cleanly, no network attempted ===")
    d = make_workdir()
    try:
        # No .env file at all in this fresh workdir.
        proc = run("setup_x_account.py", d, timeout=30)
        check("exits non-zero when .env has no credentials at all", proc.returncode != 0,
              f"exit={proc.returncode}\n{(proc.stdout or '') + (proc.stderr or '')}")
        combined = (proc.stdout or "") + (proc.stderr or "")
        check("error message mentions the missing credentials, not a generic crash",
              "X_COOKIES" in combined or "X_AUTH_TOKEN" in combined, combined)
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------
# 5: run_pipeline.py's run_step() - exit-3 SKIPPED convention + full
#    error logging for a real (asterisk-banner-shaped) crash
# ---------------------------------------------------------------------

def test_run_step_conventions():
    print("\n=== run_pipeline.py: run_step() exit-3 SKIPPED + full error logging ===")
    d = make_workdir()
    sys.path.insert(0, d)
    try:
        import importlib
        import run_pipeline as rp
        importlib.reload(rp)
        rp.DATA_DIR = d
        rp.STEP_ERROR_LOG = os.path.join(d, "pipeline_step_errors.log")

        # Fake "clean skip" script mimicking the new SKIP_REASON convention.
        skip_script = os.path.join(d, "fake_skip.py")
        with open(skip_script, "w") as f:
            f.write(
                "import sys\n"
                "print('doing some work...')\n"
                "print('SKIP_REASON: clean_text column not found - preprocess_x_text.py has not run yet')\n"
                "sys.exit(3)\n"
            )
        result = rp.run_step(skip_script, "fake_skip")
        check("run_step reports SKIPPED for exit code 3", result["status"] == "SKIPPED", result)
        check("run_step's detail is the real SKIP_REASON text",
              "clean_text column not found" in result["detail"], result)

        # Fake crash script reproducing the EXACT real incident's shape: an
        # unhandled exception whose message is bordered by asterisk rows
        # (NLTK's LookupError format), with the actual reason line buried
        # in the middle, not at the end.
        crash_script = os.path.join(d, "fake_crash.py")
        with open(crash_script, "w") as f:
            f.write(
                "import sys\n"
                "print('English rows: 700')\n"
                "sys.stderr.write('Traceback (most recent call last):\\n')\n"
                "sys.stderr.write('  File \"x.py\", line 1, in <module>\\n')\n"
                "sys.stderr.write('LookupError: \\n')\n"
                "sys.stderr.write('*' * 70 + '\\n')\n"
                "sys.stderr.write('  Resource punkt_tab not found.\\n')\n"
                "sys.stderr.write('*' * 70 + '\\n')\n"
                "sys.exit(1)\n"
            )
        result = rp.run_step(crash_script, "fake_crash")
        check("run_step reports FAILED for a real crash", result["status"] == "FAILED", result)
        check("run_step's short detail is NOT a row of asterisks",
              set(result["detail"].strip()) != {"*"}, result)
        check("run_step's short detail names the actual resource",
              "punkt_tab" in result["detail"] or "Resource" in result["detail"], result)

        check("full crash output was persisted to pipeline_step_errors.log",
              os.path.exists(rp.STEP_ERROR_LOG))
        if os.path.exists(rp.STEP_ERROR_LOG):
            with open(rp.STEP_ERROR_LOG, encoding="utf-8") as f:
                log_text = f.read()
            check("pipeline_step_errors.log contains the FULL original traceback",
                  "Resource punkt_tab not found" in log_text and "Traceback" in log_text,
                  log_text[:300])
    finally:
        sys.path.remove(d)
        sys.modules.pop("run_pipeline", None)
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    test_preprocess_normal_case()
    test_preprocess_zero_english_rows()
    test_preprocess_unicode_console_safety()

    test_downstream_script("extract_actor_mentions", "extract_actor_mentions.py", None, ["clean_text"])
    test_downstream_script("add_sentiment", "add_sentiment.py", None, ["clean_text/actors_mentioned"])
    test_add_sentiment_topic_relevant_scoping()
    test_extract_actor_mentions_topic_relevant_scoping()
    test_downstream_script("build_daily_timelines --x-only", "build_daily_timelines.py", ["--x-only"],
                            ["actors_mentioned"])
    test_downstream_script("relevance_and_relations", "relevance_and_relations.py", ["--limit", "1"],
                            ["clean_text/actors_mentioned"], extra_env={"GROQ_API_KEY": "dummy-offline-test-key"},
                            normal_case_needs_live_network=True)

    test_build_network_v2()
    test_build_network_v2_unicode_console_safety()
    test_extract_topics_hashtags_unicode_console_safety()
    test_extract_topics_hashtags_topic_relevant_scoping()
    test_clean_x_data_normal_case()
    test_clean_x_data_preserves_relevance_columns()
    test_append_relations_dedup_safeguard()
    test_fetch_x_data_diagnostics_log()
    test_fetch_actor_pagination()
    test_x_auth_cookie_loading()
    test_x_auth_build_client_loads_cookies_onto_real_client()
    test_x_auth_cookie_jar_persistence()
    test_x_auth_is_logged_in_with_retry()
    test_setup_x_account_missing_credentials_exits_cleanly()
    test_run_step_conventions()
    test_ensure_nltk_resources_unit()
    test_manual_unzip_fallback()

    print(f"\n{'='*70}\n{len(PASS)} passed, {len(FAIL)} failed\n{'='*70}")
    if FAIL:
        print("FAILED:")
        for name in FAIL:
            print(f"  - {name}")
        sys.exit(1)
    print("ALL TESTS PASSED")

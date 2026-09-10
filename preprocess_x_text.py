"""
preprocess_x_text.py

Phase 5 - text preprocessing, scoped to the 768 rows where
detected_language == "en" in x_data_cleaned.csv. Adds:

  clean_text               - lowercased, URLs/@mentions/RT-artifacts
                              removed, hashtag WORDS kept (only the '#'
                              symbol is stripped) since Phase 6 needs them.
  clean_text_lemmatized     - clean_text tokenized, English stopwords
                              removed, lemmatized (NLTK WordNet), space-
                              joined back into a string.
  long_form_excerpt         - for long_form=True rows only: first 100
                              words of clean_text (short version for
                              charts/tables that can't fit the full text).

Non-English rows are left completely untouched (all new columns blank
for them) - this script only ever narrows to language == "en" for
processing, never drops or reorders any row.

The original "text" column is never modified. Per the user: rows with
confirmed_flag == True (the 12 manually-confirmed slur/harassment hits
from the flagged-content review) should be excluded from any quoting or
display - this script doesn't touch that column, just carries it through.

Run:
    python preprocess_x_text.py

--- NLTK data self-heal (added after a real scheduled-run failure) ---
word_tokenize() and WordNetLemmatizer both need NLTK's downloaded data
files (punkt_tab, wordnet, stopwords) - separate from the `nltk` PACKAGE
itself being importable (pip install nltk is not enough on its own).
These data files live under a per-interpreter/per-profile search path
(see nltk.data.path), so on a machine with more than one Python install
(this project's own check_dependencies.py already documents that gotcha
for pip packages - the same thing applies to NLTK's data files), it's
entirely possible for `import nltk` to succeed under BOTH interpreters
while only ONE of them actually has the data downloaded.

This is exactly what caused a real scheduled-pipeline failure: manual
runs (under one Python install, with the data already downloaded) worked
fine, but the same script crashed under Task Scheduler once its "which
python.exe" path issue was fixed and it started actually running under a
different interpreter that had never had `nltk.download(...)` run for it.
The crash was an unhandled `LookupError` from word_tokenize() - NLTK
formats that error as a multi-line banner bordered by rows of '*'
characters, which is why the pipeline's status log showed a line of
asterisks instead of a real message (see run_pipeline.py's run_step() -
it used to only keep the LAST output line for its short summary, and
that banner's last line is itself a row of '*'; run_pipeline.py has
since been fixed to also save the FULL output to pipeline_step_errors.log
on any real failure, and to prefer a real "SomeError:" line over a
trailing banner/hint line for the short summary).

ensure_nltk_resources() below tries to close this gap ahead of time: for
each resource this script actually uses, check whether it's already
available; if not, attempt a quiet auto-download (this machine has normal
internet access, unlike the sandboxed bridge environment used earlier in
this project's development, where nltk.download() was blocked by an SSRF
proxy - so auto-download is expected to work here). Only if that ALSO
fails (no network, a corporate/school firewall, etc.) does this script
give up - and even then, it exits CLEANLY with a single clear
"SKIP_REASON: ..." line and exit code 3, which run_pipeline.py recognizes
as a graceful skip (same philosophy as the existing Groq-quota-wall
handling in relevance_and_relations.py): the rest of the pipeline keeps
going on whatever data already exists, and this step retries next cycle,
instead of crashing with a multi-line traceback that cascades into
KeyErrors in every step downstream that expects clean_text to exist.
"""

import os
import re
import sys
import zipfile

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Real scheduled-run incident: this script crashed with UnicodeEncodeError
# ("'charmap' codec can't encode character '\U0001f6a8'...") while printing
# a debug excerpt of a tweet containing an emoji - AFTER clean_text had
# already been computed and saved successfully. Windows' console defaults
# to a single-byte codepage (cp1252 here) that can't represent most
# emoji/non-Latin characters scraped X/Twitter text is full of, so any
# print() of raw post text is one stray emoji away from crashing the whole
# process on Windows specifically (this was never hit before simply
# because no earlier run's example-per-cluster excerpt happened to contain
# a non-cp1252 character). The real damage: run_pipeline.py then logs this
# step as FAILED even though its actual output (the CSV write) was fine -
# misleading, though harmless to downstream steps since they read the file
# from disk, not this process's exit code. Reconfiguring stdout/stderr to
# UTF-8 with errors="replace" (Python 3.7+) makes every print() in this
# script safe regardless of console codepage - unprintable characters are
# swapped for a placeholder instead of raising. Guarded in case stdout/
# stderr isn't a real reconfigurable stream in some exotic invocation
# (e.g. already redirected to something without that method).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

INPUT_CSV = "x_data_cleaned.csv"
OUTPUT_CSV = "x_data_cleaned.csv"  # updated in place, adding columns

LONG_FORM_EXCERPT_WORDS = 100

URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")
RT_RE = re.compile(r"\brt\b\s*:?", re.IGNORECASE)
HASHTAG_SYMBOL_RE = re.compile(r"#(\w+)")  # drop '#', KEEP the word
WHITESPACE_RE = re.compile(r"\s+")

# (nltk.data.find() path, nltk.download() resource name) - the find-path is
# what nltk.tokenize/nltk.corpus actually look up; the download name is
# what nltk.download() expects, and they're not always the same string.
REQUIRED_NLTK_RESOURCES = [
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
]


def _try_manual_unzip(find_path):
    """Fallback for a real quirk hit on the actual production machine:
    `python -m nltk.downloader <name>` reports success ("already
    up-to-date"), and a genuinely valid <name>.zip IS sitting on disk
    under nltk_data - but nltk.data.find() still raises LookupError,
    because the zip was never actually extracted and nltk's own
    auto-load-from-zip path didn't kick in. Reproduced directly: build a
    valid zip in a fresh nltk_data dir, nltk.data.find() fails on it every
    time, extracting it by hand immediately fixes find() - so this
    function does exactly that extraction ourselves, rather than trusting
    nltk to do it. Searches every directory already in nltk.data.path for
    "<find_path>.zip" and extracts it next to itself if found. Returns
    True if an extraction was attempted (caller should retry
    nltk.data.find() afterward), False if no matching zip exists anywhere
    (a genuinely different problem - no network, wrong account, etc.)."""
    for base in nltk.data.path:
        zip_path = os.path.join(base, *find_path.split("/")) + ".zip"
        if os.path.exists(zip_path):
            try:
                with zipfile.ZipFile(zip_path) as z:
                    z.extractall(os.path.dirname(zip_path))
                return True
            except Exception as exc:
                print(f"  found {zip_path} but couldn't extract it: {exc}")
    return False


def ensure_nltk_resources():
    """Makes sure every NLTK data file this script needs is actually
    present for THIS interpreter, auto-downloading anything missing. See
    the module docstring's "NLTK data self-heal" section for why this
    exists. Returns silently if everything's fine (the common case, and
    silent - no need to spam normal-case output). Exits the whole process
    cleanly (SKIP_REASON + exit code 3) if a resource is missing and
    cannot be made available - never lets an unhandled LookupError
    propagate and cascade into confusing KeyErrors in every step
    downstream."""
    missing = []
    for find_path, download_name in REQUIRED_NLTK_RESOURCES:
        try:
            nltk.data.find(find_path)
        except LookupError:
            missing.append((find_path, download_name))

    if not missing:
        return

    still_missing = []
    for find_path, download_name in missing:
        print(f"NLTK resource '{download_name}' not found locally - attempting auto-download...")
        try:
            ok = nltk.download(download_name, quiet=True)
        except Exception as exc:
            ok = False
            print(f"  auto-download of '{download_name}' raised: {exc}")

        found = False
        if ok:
            try:
                nltk.data.find(find_path)
                print(f"  OK - '{download_name}' downloaded successfully.")
                found = True
            except LookupError:
                pass

        if not found:
            # Either the download itself failed, OR (a real case we've
            # hit) it reported success / "already up-to-date" but the zip
            # was never extracted - try extracting it ourselves before
            # giving up. See _try_manual_unzip()'s docstring.
            if _try_manual_unzip(find_path):
                try:
                    nltk.data.find(find_path)
                    print(f"  OK - '{download_name}' found after manually extracting its zip file.")
                    found = True
                except LookupError:
                    pass

        if not found:
            still_missing.append(download_name)

    if still_missing:
        names = ", ".join(still_missing)
        print(
            f"SKIP_REASON: NLTK resource(s) [{names}] are not available and could not be "
            f"auto-downloaded (no network access, or a firewall is blocking nltk.download - "
            f"this can also happen if this run is under a DIFFERENT Python install than the "
            f"one you normally use manually, per this project's check_dependencies.py notes "
            f"about multiple Python installs). Fix by running once, with the SAME interpreter "
            f"Task Scheduler uses: python -m nltk.downloader {' '.join(still_missing)}"
        )
        sys.exit(3)


# Must run BEFORE the module-level STOPWORDS line below, which itself
# touches NLTK data (stopwords.words(...)) at IMPORT time - if this call
# were deferred to inside main() instead, a missing 'stopwords' corpus
# would crash on the STOPWORDS line below before main() (and this
# self-heal check) ever ran.
ensure_nltk_resources()

STOPWORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def make_clean_text(text):
    t = (text or "").lower()
    t = URL_RE.sub("", t)
    t = MENTION_RE.sub("", t)
    t = RT_RE.sub("", t)
    t = HASHTAG_SYMBOL_RE.sub(r"\1", t)  # "#makkahpact" -> "makkahpact"
    t = WHITESPACE_RE.sub(" ", t).strip()
    return t


def lemmatize_clean_text(clean_text):
    tokens = word_tokenize(clean_text)
    kept = [w for w in tokens if w.isalpha() and w not in STOPWORDS]
    lemmas = []
    for w in kept:
        # WordNetLemmatizer's default noun-POS assumption mangles some
        # short words - notably "us" -> "u" (it treats it as a plural of
        # "u"), which matters here since "US" is one of the 5 tracked
        # actors. Skip lemmatization for very short tokens; there's
        # essentially nothing useful to lemmatize at that length anyway.
        lemmas.append(w if len(w) <= 2 else lemmatizer.lemmatize(w))
    return " ".join(lemmas)


def main():
    df = pd.read_csv(INPUT_CSV, dtype={"id": str})

    en_mask = df["detected_language"] == "en"
    n_en = en_mask.sum()
    print(f"English rows (detected_language == 'en'): {n_en}")
    if n_en == 0:
        # Not a crash and not really a "skip" either - genuinely nothing
        # to do this cycle (e.g. every collected row so far failed
        # language detection). .apply() on an empty index is already a
        # harmless no-op below, so this is just here to make that state
        # visible in the log instead of looking identical to a normal run
        # that actually processed something.
        print("0 English rows to process this cycle - nothing to do, exiting normally.")

    # Initialize new columns as blank for ALL rows (non-English rows stay
    # untouched), then fill in only for the English subset.
    for col in ["clean_text", "clean_text_lemmatized", "long_form_excerpt"]:
        if col not in df.columns:
            df[col] = ""

    en_idx = df.index[en_mask]
    df.loc[en_idx, "clean_text"] = df.loc[en_idx, "text"].apply(make_clean_text)
    df.loc[en_idx, "clean_text_lemmatized"] = df.loc[en_idx, "clean_text"].apply(lemmatize_clean_text)

    long_form_en = en_idx[df.loc[en_idx, "long_form"] == True]
    df.loc[long_form_en, "long_form_excerpt"] = df.loc[long_form_en, "clean_text"].apply(
        lambda t: " ".join(t.split()[:LONG_FORM_EXCERPT_WORDS])
    )

    df.to_csv(OUTPUT_CSV, index=False)

    print(f"long_form rows within English subset: {len(long_form_en)}")
    print(f"Saved -> {OUTPUT_CSV}")

    # --- Step 6: one example per duplicate-content cluster, English subset only ---
    en_df = df.loc[en_idx]
    clustered_en = en_df[en_df["duplicate_content_cluster_id"].fillna("") != ""]
    all_clusters = df[df["duplicate_content_cluster_id"].fillna("") != ""]["duplicate_content_cluster_id"].nunique()
    clusters_in_en = clustered_en["duplicate_content_cluster_id"].nunique()
    print(f"\nDuplicate-content clusters overall: {all_clusters}; "
          f"clusters with >=1 row in the English subset: {clusters_in_en}")

    print("\n=== One example per cluster (English subset) ===")
    reviewable = clustered_en[clustered_en["confirmed_flag"] != True]  # honor the exclusion
    for cid, group in reviewable.groupby("duplicate_content_cluster_id"):
        row = group.iloc[0]
        size = (df["duplicate_content_cluster_id"] == cid).sum()
        snippet = str(row["text"]).replace("\n", " ")[:140]
        print(f"[{cid}] size={size} matched_actor={row['matched_actor']} "
              f"user=@{row['username']}: {snippet}...")


if __name__ == "__main__":
    main()

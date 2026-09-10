# Makkah Pact Pipeline — Investigation Log

Purpose: this file is the durable memory for this project's debugging history.
Conversation summaries lose detail over time; this file does not. **Any Claude
session (or you) picking this project back up should read this file first,
top to bottom, before touching code.** Append to it — don't delete history.

---

## 2026-09-09 — X/Twitter fetch failure investigation

### The symptom
`fetch_x_data.py` / `fetch_x_test.py` return 0 tweets. twscrape raises (internally,
caught by twscrape itself, never visible to our except blocks) one of:
- `XClIdParseError: X web scripts not found`
- `XClIdAccountError: Logged-out X web app`

Both come from `twscrape/xclid.py`, which fetches `x.com/tesla` using the
account's cookies to compute the `x-client-transaction-id` anti-bot header.
X is serving the **anonymous/logged-out web app bundle** to that request —
meaning X's server, not our code, is deciding this session doesn't count as
logged in, regardless of which cookies we send.

### What we already fixed this week (confirmed working, keep these)
1. **`setup_x_account.py` silent no-op bug** — `api.pool.add_account()` is a
   no-op if the account already exists in `accounts.db` (only logs a warning).
   Every cookie refresh for months silently did nothing. Fixed by switching to
   `api.pool.add_account_cookies()` (a real upsert). **Confirmed via the git
   history below: this bug existed in the code from day one** — it just never
   mattered until now, because it only bites on the *second+* setup run for
   the same account (see "why it worked originally" below).
2. **`fetch_x_data.py` / `fetch_x_test.py` never called `load_dotenv()`** —
   fixed; confirmed `TWS_HTTP_BACKEND` and other `.env` settings now actually
   reach the process.
3. **`curl_cffi` backend (`TWS_HTTP_BACKEND=curl`)** — installed, confirmed
   engaged (`backend=curl` in logs). Did **not** fix the core error — same
   `XClIdAccountError` occurred with curl backend too. Ruled out TLS
   fingerprinting as the sole cause.
4. **`X_COOKIES` full-cookie-string support** — added so the *entire* browser
   `Cookie:` header (15 cookies) can be used instead of just `auth_token`/`ct0`.
   Mechanically confirmed working (`setup_x_account.py` genuinely saved 15
   fresh cookies). Did **not** fix the core error either — the very next live
   test reverted to `XClIdParseError`. Ruled out "not enough cookies" as the
   cause.

### Git history comparison (done 2026-09-09, via direct git-object parsing —
see note on device_bash below)
Compared the **very first commit** (`33dc6d3`, when the user confirms fetching
worked perfectly with just 2 cookies) against the current `fetch_x_data.py` /
`setup_x_account.py`. Full diff was pulled and reviewed line by line.

**Finding: nothing in the fetch/auth logic changed in a way that could cause
this.** The diff against the original is purely additive:
- UTF-8 console reconfiguration (Windows crash fix, unrelated)
- `fetch_diagnostics.log` writing (visibility only, doesn't touch requests)
- `load_dotenv()` call (added this week, see fix #2 above)
No change to: the search query construction, `api.search(query, limit=...)`
call itself, delays/pacing, actor terms, or headers. The original
`setup_x_account.py` used the exact same `auth_token=...; ct0=...` two-cookie
format and the buggy `add_account()` call — **identical to what's "supposed"
to work per the user's memory.**

**Why it worked the first time despite the `add_account()` bug**: the very
first call to `add_account()` for a brand-new username is a real INSERT (the
no-op only triggers when a row already exists). So the original 2 cookies
really were saved, and they kept working via that single, never-rotated
session for as long as they stayed valid. Every *subsequent* "refresh cookies
in .env and rerun setup" cycle (which is what's been happening for weeks,
per the user) silently did nothing — the account pool kept serving whatever
that first session was, until it expired/got invalidated, at which point
every fetch would fail exactly like we've been seeing, no matter what new
cookies were pasted into `.env`, because they were never actually being
saved. **This week's `add_account_cookies()` fix is therefore the first time
in this account's life a cookie refresh has genuinely taken effect.**

### Leading hypothesis (evidence-based, not another guess-and-check)
Two non-exclusive possibilities, both consistent with every observation:

1. **Session churn is itself the trigger.** Before this week, X only ever
   saw ONE long-lived, stable session for this account (because refreshes
   were silently failing). This week, for the first time, the account has
   had its session cookies swapped out for genuinely new ones *multiple
   times in a single day* (the user's own words: "i have done this multiple
   time today"). Rapidly rotating sessions from what looks like the same
   client/pipeline is a classic automated-behavior signal — X's server-side
   heuristics may be specifically choosing to serve the sanitized
   "logged-out" bundle to this request pattern, independent of whether the
   cookies themselves are individually valid.
2. **This is a known, currently-open twscrape/X compatibility issue.**
   [GitHub issue #320](https://github.com/vladkens/twscrape/issues/320)
   describes this *exact* error and root cause ("anonymous/logged-out
   requests get served a web app version that no longer includes the
   script"), is still open with no confirmed fix as of this writing, and
   is not account-specific in its description. If X changed something
   platform-wide in how it decides a request is "logged in" recently,
   every twscrape user could be hitting this now regardless of account
   health — which would explain why cookie count / TLS backend never
   mattered.

Both point away from "our code is misconfigured" and toward "the
account's session history and/or twscrape's current compatibility with
X" — which matches the user's original, correct instinct that this
shouldn't be a simple one-line code fix.

### Recommended next steps (in order, cheapest/most diagnostic first)
1. **Stop rotating cookies repeatedly.** Every extra login-and-swap cycle
   may be reinforcing whatever suspicion X's heuristics have already formed
   about this session pattern. Do ONE clean cycle: log in once, grab cookies
   once, run `setup_x_account.py` once, then **wait at least 30–60 minutes
   without touching `.env` or accounts.db again** before testing with
   `fetch_x_test.py`. If it works after a quiet period, that confirms
   hypothesis #1 (churn-triggered).
2. **Differential test with a brand-new, never-scraped X account.** Create a
   second throwaway account, log in normally in a browser, grab just
   `auth_token`/`ct0` (the original 2-cookie method — matches what's proven
   to have worked), run `setup_x_account.py` against it, test once. If THIS
   account also gets `XClIdAccountError`/`XClIdParseError` immediately, that
   is strong evidence of hypothesis #2 (library/X-wide issue, not this
   account) — and the fix isn't in our code at all, it's a twscrape-side
   problem to work around or route around (see step 3).
3. **If step 2 also fails**, the pragmatic, deadline-driven move is to stop
   debugging twscrape further and pivot: either (a) X's official API (paid,
   but not subject to this anti-bot cat-and-mouse), or (b) try `twikit`
   (another cookie-based scraping library, actively mentioned as a current
   working alternative) as a swap-in for the fetch layer only — everything
   downstream (cleaning, relevance, relations, dashboard) is unaffected
   either way since they only consume `x_data.csv`.

## 2026-09-10 — Step 2 result: CONCLUSIVE, this is not account-specific

Ran step 2 (differential test) **before** step 1 (the quiet-period test on the
old account) — user created a brand-new X account (phone-verified signup,
email alias linked after), used ITS cookies (first-ever use, no churn
history — same "clean first use" pattern that worked on the original
account back in the initial commit), ran `setup_x_account.py` (succeeded,
`logged_in=True active=True`), ran `fetch_x_test.py` **immediately** (no
wait needed here — see reasoning below). Result: **identical failure**,
`XClIdParseError: X web scripts not found`, `backend=curl` confirmed.

(Note: `setup_x_account.py`'s `ACCOUNT_LABEL` constant is hardcoded to
`"makkah_pact_x_scraper"` on purpose — see the script's own docstring — so
the error log's `username=makkah_pact_x_scraper` is just the local pool
nickname, not the real X account. The cookies underneath genuinely are the
new account's. This is expected, not a mix-up.)

**This rules out hypothesis #1 (session-churn/account-specific flagging)
and confirms hypothesis #2: this is a platform-wide, currently-open,
unresolved twscrape↔X compatibility break, not fixable from our side.**

Corroborating evidence found via web research immediately after:
[GitHub issue #330](https://github.com/vladkens/twscrape/issues/330)
("XClIdParseError ... Cloudflare challenge shell served instead of app
HTML"), opened 2026-08-25 by another user, describes this exact symptom in
forensic detail: authenticated requests to `x.com/tesla` (valid
`auth_token`+`ct0`, curl_cffi Chrome impersonation) get served a **legacy
webpack build carrying Cloudflare's own challenge-platform scripts**
(`/cdn-cgi/challenge-platform/scripts/jsd/api.js`) instead of the expected
x-web/Vite bundle that contains the signing script twscrape needs. The
reporter explicitly states it's **"reproducible without proxies and across
multiple authenticated accounts."** twscrape's latest release (0.20.1) is
dated the SAME DAY this issue was opened (2026-08-25) — no patch has
shipped addressing it in the ~16 days since. This is Cloudflare sitting in
front of X, not X's own account-level session validation — which is why
neither more cookies, nor curl_cffi TLS impersonation, nor a fresh account
made any difference: none of those address a Cloudflare bot-challenge.

### Decision: stop debugging twscrape's cookie/account layer. Pivot the fetch library.
Given the deadline, continuing to vary cookies/accounts/backends against
twscrape is no longer productive — it's a confirmed upstream bug with no
ETA. Researched current alternatives:
- **twikit** — popular (4.2k★), cookie-based, but has had its own recent
  breakage reports against 2026-era X/Cloudflare changes.
- **twifork** (https://github.com/PawiX25/twifork) — an actively maintained
  fork of twikit, explicitly built to patch exactly this class of 2026
  breakage: fixes a missing `frame_time` rounding step in
  `x-client-transaction-id` generation (their issues #357/#397) and offers
  built-in `curl_cffi` Chrome impersonation (`impersonate='chrome124'`) for
  requests that get 403'd on the default httpx fingerprint. Cookie-based
  auth only (`client.set_cookies()`/`load_cookies()` — matches what we
  already have, no new credential-gathering needed), drop-in API-compatible
  with twikit (`from twikit import Client` still works per its README).
  This is the recommended next step: swap `fetch_x_data.py`,
  `fetch_x_test.py`, and `setup_x_account.py`'s twscrape calls for
  twifork's client, keeping everything downstream of `x_data.csv`
  untouched. Proposed to the user 2026-09-10, awaiting go-ahead before
  writing code (per the user's own "plan properly before changing things"
  instruction, and the standing dependency-check rule before adding a new
  import).

## 2026-09-10 — Pivot implemented: twscrape -> twifork/twikit

Rewrote the fetch layer per the decision above. Changes, all deployed to
the real project folder and verified before deploying:

- **New file `x_auth.py`** - shared cookie-loading + `Client` construction.
  Reads `.env` (X_COOKIES or X_AUTH_TOKEN/X_CT0 fallback, same precedence
  as before), converts to the `{name: value}` dict twikit's `set_cookies()`
  wants (twscrape took a raw string - this is the one real format
  difference), builds a `Client("en-US", impersonate="chrome124")`.
- **`setup_x_account.py`** - rewritten. No more `accounts.db`/account pool
  (twikit's `Client` works directly off cookies each run - there's nothing
  to "add" to). Now just builds a client and calls the real
  `client.is_logged_in()` (an actual network check against X, not a local
  flag) to confirm the session works before you bother running the fetch
  scripts.
- **`fetch_x_test.py` / `fetch_x_data.py`** - rewritten to use
  `client.search_tweet(query, "Latest", count=...)` with real page-boundary
  pagination via the returned `Result`'s `.next()` (twikit hands back
  exactly `count` tweets per page - a REAL boundary, unlike twscrape's
  approximated one). Field mapping: `t.rawContent`->`t.full_text`,
  `t.user.username`->`t.user.screen_name`, `t.date`->`t.created_at_datetime`,
  `t.likeCount`->`t.favorite_count`, `t.retweetCount`->`t.retweet_count`.
  Everything else (actor list, topic terms, pacing constants, DIAG_LOG,
  `--recent` mode, `NEW_ROWS_ADDED` line, output CSV columns) is
  byte-for-byte unchanged - downstream scripts need no changes.
- **`requirements.txt`** - `twscrape`/`curl_cffi` replaced with
  `twifork[impersonate]==2.4.0` (pinned - unlike `twscrape`, which was left
  unpinned the whole project and silently drifted underneath us; don't
  repeat that here without a deliberate reason to upgrade).
- **`test_missing_columns_offline.py`** - the old twscrape/accounts.db-
  specific tests (`test_setup_x_account_cookie_refresh`,
  `test_setup_x_account_full_cookie_string`, the two
  `*_loads_dotenv_before_api_client` tests) were replaced with equivalents
  for the new architecture (`test_x_auth_cookie_loading`,
  `test_x_auth_build_client_loads_cookies_onto_real_client`,
  `test_fetch_actor_pagination`, `test_setup_x_account_missing_credentials_exits_cleanly`).
  Full suite: **81/81 passing**, run against a genuinely fresh venv with
  `twifork[impersonate]==2.4.0` actually installed from the updated
  `requirements.txt` (not just mocked) - `x_auth.build_client()` was
  verified end-to-end against the real installed `twikit` package
  (cookies set via `client.set_cookies()` read back correctly via
  `client.get_cookies()`), and `fetch_actor()`'s new pagination logic was
  verified against a fake paginated client for both the "stop exactly at
  the limit, even mid-page" and "stop cleanly on an empty page" cases.

**Why twifork specifically** (not the original `twikit`): reading its
actual source (not just the README) turned up two direct, deliberate fixes
for the exact failure class we hit - its `is_logged_in()` explicitly
treats a non-JSON 200 response (a Cloudflare interstitial - precisely what
issue #330 describes) as "not logged in" instead of crashing, and
`get_cookies()` has a comment about handling X's duplicate `__cf_bm`
cookie (Cloudflare's own bot-management cookie) across domains without
dying on a `CookieConflict`. That's real, recent, targeted hardening
against this project's exact symptom, not just marketing copy.

**What could NOT be verified offline** (honest limitation, not glossed
over): whether this actually gets past X/Cloudflare and returns real
tweets. `is_logged_in()` and `search_tweet()` both make genuine network
calls by design, so that can only be confirmed by an actual run on the
real machine. **This has NOT been run by Claude - the user needs to run it
themselves next**:
```
pip install -r requirements.txt
python setup_x_account.py
python fetch_x_test.py
```
`accounts.db` is no longer read or written by any of these scripts - safe
to ignore, not deleted automatically.

## 2026-09-10 — CONFIRMED WORKING on the real device

User ran the pivot for real:
```
python setup_x_account.py   -> "Logged in successfully."
python fetch_x_test.py      -> "-> 10 tweets returned", real on-topic
                                tweets dated 2026-09-09, saved to
                                x_test_sample.csv
```
This is the actual fix - not just mechanically-correct-but-still-broken
like every twscrape attempt this week. The twscrape->twifork pivot (see
the two sections above) resolved the real, months-long collection outage.

Only cosmetic byproduct, safe to ignore: a `RuntimeWarning` from
`curl_cffi`'s asyncio integration on Windows ("Proactor event loop does
not implement add_reader... registering an additional selector thread").
curl_cffi handles it automatically (the warning says so) - it's not an
error and didn't affect the result. If the noise is annoying, the fix
would be setting `asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())`
at the top of each script, but this is optional polish, not a bug.

### Next step: full historical backfill
`fetch_x_test.py` only confirms login + a 10-tweet smoke test. The real
collection gap is still open - recall the 16-day gap identified earlier
between the last real data in `x_data.csv` and "today". Next: run the full
pipeline (`python run_pipeline.py`, or `run_pipeline.bat`) so
`fetch_x_data.py` does its full backfill pass (since 2026-08-03, up to 300
tweets/actor across all 5 actors) and every downstream step (cleaning,
relevance/relations via Groq, network build, dashboard data) catches up
behind it in one run - not just `fetch_x_test.py` again, which only ever
touches `x_test_sample.csv`, a throwaway file the rest of the pipeline
doesn't read.

## 2026-09-10 — RESOLVED: full pipeline run succeeded end to end

User ran `python run_pipeline.py` (the full historical backfill + full
downstream pipeline). Result: **9/9 steps SUCCESS**, ~33 minutes total:

```
[1/9] fetch_x_data              SUCCESS (93s)   - 122 new row(s)
[2/9] clean_x_data              SUCCESS (76s)
[3/9] preprocess_x_text         SUCCESS (27s)
[4/9] extract_topics_hashtags   SUCCESS (4s)
[5/9] extract_actor_mentions    SUCCESS (2s)
[6/9] add_sentiment             SUCCESS (7s)
[7/9] relevance_and_relations   SUCCESS (1767s)  <- Groq relevance/relation
                                                     extraction, ~29 min,
                                                     expected for a batch
                                                     this size, not a hang
[8/9] build_network_v2          SUCCESS (4s)
[9/9] build_daily_timelines     SUCCESS (2s)
```

122 new rows fetched on the first real collection run since the account's
session broke (weeks of the `add_account()` no-op bug, then the
twscrape/Cloudflare outage documented above). **This closes out the whole
investigation** - the original ask ("get scraping working again") is done.

### Status for whoever reads this next
The X/Twitter collection pipeline is WORKING as of 2026-09-10, on
`twifork==2.4.0` (see the two sections above for why and how). If it
breaks again later, don't restart from cookie/account theories - check
`fetch_diagnostics.log` and `pipeline_status.log` first, and if it's the
`XClIdParseError`/`XClIdAccountError`/Cloudflare-shell symptom again,
check https://github.com/PawiX25/twifork for a newer release before
re-investigating from scratch - this exact class of breakage has hit
twscrape twice in one month (issues #320, #330) and may recur upstream in
twifork too as X keeps changing its anti-bot posture.

### Known loose end (not yet acted on)
`.git/index.lock` (0 bytes) exists in the project's `.git` folder — a stray
lock file, likely from an interrupted git operation. This can make the
user's own local `git` commands fail with "Unable to create .git/index.lock:
File exists" until it's deleted. Safe to delete manually if you hit that
error (it's just a lock marker, not real data).

### Operational note: device_bash was down during this investigation
The direct shell bridge to the user's machine (`device_bash`) returned
`sandbox-helper: no Plan9 drive shares mounted` on every command this
session, including trivial ones — a transient bridge issue, not a project
bug. Worked around it by reading the `.git/objects/*` files directly via
`device_list_dir`/`device_stage_files` and parsing the git object format
(zlib-decompressed `commit`/`tree`/`blob` objects) in the cloud sandbox with
a small Python script, rather than running `git show`. If `device_bash`
comes back, prefer normal `git` commands going forward — this was a fallback.

---

## 2026-09-10 — relevance_and_relations timeout diagnosis + full codebase cleanup audit

### Part 1: why `relevance_and_relations` failed with "timed out after 2700s"

**Verdict: Groq free-tier rate-limit pacing wall for an unusually large batch, not a code bug.** No code changes were needed or made for this part.

- The run in question followed a 543-row `fetch_x_data.py` fetch (the biggest single batch since the original historical backfill) — every one of those rows starts with `topic_relevant = NaN`, so all 543 needed the expensive "full" classification pass (relevance + relations), not just the cheaper "relations only" pass.
- `relevance_and_relations.py` paces itself against Groq's real limits: `TPM_LIMIT=8000` and a tighter, undocumented `OTPM_LIMIT=1000` (output-tokens-per-minute), both derated by a 0.85 safety factor, with `PER_POST_COMPLETION_CEILING_FULL=450` tokens/post and `BATCH_SIZE_DEFAULT=3` posts/call. Worst case, that's roughly **1 batch of 3 posts per minute** — around 3 rows/minute.
- At that rate, 543 new rows can plausibly take 90-180+ minutes. `run_pipeline.py`'s `RELATIONS_TIMEOUT_SEC = 45 * 60` (2700s) is already deliberately generous (its own comment says so), but a batch this size still exceeds it.
- Confirmed via `pipeline_status.log`: this exact run (`RUN START` 2026-09-10T00:03:28Z, 3088s total) is the one in the pasted terminal output, and a prior run on 2026-09-08 timed out the same way for the same reason.
- **No progress was lost.** `process_group()` calls `save_checkpoint(df)` and `append_relations()` after every batch (not just at the end), so a timeout-kill only stops future batches — everything already classified is saved. The pipeline will keep chipping away at the backlog on subsequent scheduled runs until it's caught up; no manual intervention needed.
- **One small, real observability gap found, not fixed (optional future improvement):** when the process is killed on timeout, its progress-checkpoint `print()` output is lost rather than logged to `pipeline_step_errors.log`, because Python fully block-buffers stdout when captured via a pipe and `relevance_and_relations.py`'s `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` call doesn't set `line_buffering=True`. Low priority — the checkpoint file itself (not the printed log) is the real source of truth and was never at risk — but worth a one-line fix later (`line_buffering=True` on that reconfigure call, or an explicit `flush()` after each checkpoint print) so the error log stops silently coming up empty on this exact failure mode.

### Part 2: full codebase cleanup audit

Full read-through and cross-reference of every file in the project folder against `app.py`'s actual read list, `run_pipeline.py`'s actual call chain, and each file's own docstrings/comments. Nothing was deleted or changed without first confirming it via the code itself (not guessed).

**Confirmed 100% dead code — safe to delete, but I could NOT delete these myself this session** (see "why files weren't deleted" below):
- `build_actor_network.py` — reads legacy `actor_frequency_deduped.csv`/`actor_cooccurrence_deduped.csv`, writes `actor_network.json`. `run_pipeline.py`'s own docstring says this was "Deliberately DROPPED... Both only ever fed the OLD actor_network.json, which the dashboard no longer reads."
- `dedupe_actor_cooccurrence.py` — only ever fed the file above. Same docstring confirms it's dropped.
- `refresh_worker.py` — old background-subprocess chain launched by app.py's "Refresh Data" button. `app.py`'s own docstring says that button "has been removed entirely, on purpose."
- Their orphaned output files, confirmed not in `app.py`'s read list: `actor_network.json`, `actor_frequency_deduped.csv`, `actor_cooccurrence_deduped.csv`.
- `nul` (53 bytes) and `run_pipeline_manual_test.log` (0 bytes) — Windows shell-redirect artifacts, zero references anywhere in the codebase.

*Why it's safe:* `app.py` reads only `x_data_cleaned.csv`, `actor_network_v2.json`, `actor_centrality.csv`, `daily_actor_timeline.csv`, `daily_gdelt_timeline.csv`, `hashtag_frequency.csv`, `pipeline_status.log` — none of the files above are on that list, and `run_pipeline.py`'s current 9-step chain never calls any of these three scripts.
*Impact of removing:* none on the running app; only removes confusion for anyone reading the folder fresh.
*Risk:* essentially none — confirmed dead via the project's own documentation, not inference.

**Not read by `app.py`, but still legitimately produced each cycle — no action needed, not legacy:** `actor_frequency.csv`, `actor_cooccurrence.csv` (from `extract_actor_mentions.py`), `hashtag_cooccurrence.csv` (from `extract_topics_hashtags.py`). These just aren't surfaced in the UI yet; they're current pipeline output, not dead code.

**Needs your decision, not auto-deleted:** `x_data_cleaned_BACKUP_2026-09-08.csv` and `actor_relations_BACKUP_2026-09-08.csv`. Nothing reads them, but the deliberate "BACKUP" naming from a specific past incident means I flagged rather than deleted — your call whether you still want that safety net.

**Two real, small bugs found and fixed this session (deployed):**
1. `app.py`'s Sentiment & Engagement tab had a stale empty-data warning referencing a "🔄 Refresh Data" button that no longer exists (the button was renamed to "🔄 Check for updates" when the refresh flow moved into `run_pipeline.py`). Fixed to reference the real button name.
2. `start_dashboard_and_share.bat` line 38 had the tunnel command still pointed at the literal unfilled placeholder `YOUR-DEV-DOMAIN-HERE`, while the console message on line 42 showed the real domain (`dioxide-false-declared.ngrok-free.dev`). This meant the actual ngrok tunnel would have failed even though the printed message looked correct. Fixed line 38 to use the real domain, matching line 42.

**Confirmed necessary, keep as-is, no action:** `check_dependencies.py` (standalone diagnostic for a recurring multi-Python-install issue), `run_pipeline.bat` (the real Task Scheduler entry point), `fetch_gdelt.py` (intentional standalone one-time GDELT backfill script), `flagged_review.csv` (live, human-maintained moderation input that `clean_x_data.py` reads every cycle — never delete), `test_relevance_and_relations_offline.py` (distinct test coverage from `test_missing_columns_offline.py`, not a duplicate), `project_brief_handoff.md` (the original professor-approved brief — different purpose from this log). All 14 packages in `requirements.txt` were grep-confirmed to actually be imported somewhere; no dead dependencies.

**Flagged technical debt, NOT executed (deliberately — see reasoning below):** the actor list `ACTORS = ["Iran", "Pakistan", "Saudi Arabia", "United States", "Turkiye"]` is duplicated identically across 6 files (`add_sentiment.py`, `app.py`, `extract_actor_mentions.py`, `build_daily_timelines.py`, `dedupe_actor_cooccurrence.py` — about to be deleted, `extract_topics_hashtags.py`), and `ACTOR_KEYWORDS` is duplicated across 2 files, both with comments saying "kept in sync on purpose." Recommended fix: extract both into one shared `constants.py` imported everywhere. Not done now because it touches 6+ files right before your deadline for a cosmetic/maintainability win with real (if small) regression risk — worth doing after submission, not during this cleanup pass.

**Why the 8 confirmed-dead files weren't deleted this session:** `device_bash` (the direct command channel to your machine) is down again this session — it returns `"sandbox-helper: no Plan9 drive shares mounted"` on every command, a known/recurring bridge issue, not a project bug. File *reads and writes* to your machine still work fine (that's how the two bug-fixed files above got delivered), but deletion has to run as a real shell command, which this bridge issue blocks. These 8 files need to be deleted by you directly — see the message accompanying this log update for the exact list and easiest way to do it.

---

## 2026-09-10 — automated GitHub sync for the Streamlit Cloud deployment

### What was already there
`run_pipeline.py` already contained a complete, working `sync_to_git()` step (gated behind `ENABLE_GIT_SYNC = False`) from earlier work on this project - `git add` the exact files `app.py` reads (`SYNC_FILES`) -> `git commit` -> `git push`, with "nothing to commit" already treated as a clean no-op SUCCESS. It authenticated via a plain `git push`, relying on a one-time manual login caching credentials in Windows' Git Credential Manager. That's a legitimate approach, but not what was asked for this round.

### What changed
Per your request, rewrote the authentication mechanism to use a GitHub Personal Access Token read from `.env` (`GITHUB_PAT=...`) instead of a cached credential, so the push step is fully self-contained and doesn't depend on a prior interactive login ever having happened on this machine:
- `sync_to_git()` now reads `GITHUB_PAT` from `.env` at call time (`load_dotenv()`, matching this project's existing per-call pattern from `x_auth.py`). If it's missing, the step logs a clean `SKIPPED` (not a crash, not a `FAILED`) and does nothing else - no commit is even attempted.
- The authenticated push URL (`https://<PAT>@github.com/<owner>/<repo>.git`) is built in memory from `git remote get-url origin` plus the token, and passed directly as a `git push` argument - it is never written to `.git/config` (no `git remote set-url`), so it never appears in a later `git remote -v` and never persists anywhere on disk.
- Any git output that might echo the token back (some git versions include the full remote URL, credentials and all, in their own error/hint text) is passed through a `_redact()` regex before it's ever printed or logged. Verified by test that the token never appears in any logged/returned string, including on a failed push.
- "Only push if something changed" was already effectively true via git's own "nothing to commit" detection (push is structurally unreachable if commit was a no-op) - kept as-is, no separate `git diff` pre-check needed.
- Commit message format: `"Automated data refresh (<N new rows, if any>) - <UTC timestamp>"`.
- Failure handling (bad/expired PAT, network issue, rejected/diverged push) all fall through to the same graceful `FAILED` path already established elsewhere in this pipeline - logged with a short, redacted, informative reason, never crashes the run, never blocks the local file writes that already happened earlier in the cycle.
- Flipped `ENABLE_GIT_SYNC` to `True` (it was `False` because there was no Streamlit Cloud deployment to push to yet - there is now). **Known, expected interim state:** until `GITHUB_PAT` is actually added to `.env`, every pipeline run will show `sync_to_github: SKIPPED` and the run's overall status in `pipeline_status.log` will read "partial" instead of "success" - purely because of this one step. Not a bug; clears up the moment the PAT is added.
- Updated the module docstring's "Prerequisite" section (previously described the old GCM-cache login flow, now describes the PAT/`.env` flow) so it stays accurate.

### How it was tested (offline, before touching the real repo)
`test_git_sync_offline.py` (new, kept as a permanent offline test alongside the project's other `test_*_offline.py` files) imports the real `sync_to_git()` from `run_pipeline.py` itself - not a reimplementation - and runs it against a local bare git repo standing in for GitHub. The real push URL the code builds (`https://<token>@github.com/owner/repo.git`) is transparently redirected to the local bare repo via git's own `url.<path>.insteadOf` rewrite mechanism, so the exact real code path (URL construction, the actual `git push` subprocess call, real git success/failure responses) is exercised end-to-end with zero network/GitHub involvement. 5 scenarios, 16 assertions, all passing:
1. `GITHUB_PAT` unset -> clean `SKIPPED`, confirmed no commit/push was even attempted.
2. `GITHUB_PAT` set, nothing changed -> `SUCCESS`, confirmed no-op (bare repo untouched).
3. `GITHUB_PAT` set, real change -> `SUCCESS`, confirmed the bare repo actually received the new commit, confirmed the commit message carries the new-row count, confirmed the fake token never appears anywhere in the returned result.
4. Push rejected (a second clone pushes a divergent commit first, simulating "someone/something else touched the repo") -> `FAILED` gracefully, no exception, no token leak.
5. Origin remote not recognized as a `github.com` URL -> `FAILED` gracefully with a clear reason, no crash.

Also confirmed: `python-dotenv` (the only new import) is already in `requirements.txt`; `py_compile` clean; a fresh venv import of the edited `run_pipeline.py` succeeds; no existing offline test file references `run_pipeline.py`, so nothing else could have been broken by this change.

### What you still need to do
Generate a GitHub PAT and add it to `.env` as `GITHUB_PAT=...` - see the setup steps delivered alongside this log update. Until that line is added, `sync_to_github` will show as a harmless `SKIPPED` step every run.

---

## 2026-09-10 (later) — live-link audit against professor feedback + dashboard redesign

### Live deployment audit
Checked `makkah-pact-analysis.streamlit.app` directly (browser) and the GitHub repo behind it. Finding: the repo has exactly **one commit ever**, from before the LLM relation-extraction rewrite and the topic-relevance scoping fix - meaning the live dashboard was still showing the old co-occurrence-only actor network and unscoped sentiment data, i.e. exactly what the professor's original feedback flagged, even though both were already fixed locally. User confirmed the professor's verbatim original note, which maps directly onto: (1) "use LLMs for inference and relations" - already addressed via `relevance_and_relations.py` + `build_network_v2.py`; (2) "hashtags are not relevant or in line with mecca pact" - already addressed via `topic_relevant` scoping; (3) "correlated for sentiments and analysis" - previously only a visual GDELT overlay and side-by-side bar charts, no computed statistic (addressed below).

`.gitignore` was tightened (added `x_data.csv`, `x_test_sample.csv`, `*_BACKUP_*.csv`, and the local-only diagnostic logs - `pipeline_status.log` deliberately left trackable since it's in `SYNC_FILES`) and deployed, so a plain `git add -A` is now safe for the user's one-time catch-up push bringing GitHub current with local - exact commands given to the user directly (not run by Claude; `device_bash` still unavailable this session).

### Dashboard redesign (`app.py`)
Per explicit user request: added the at-a-glance summary strip, computed correlation statistics, consistent actor colors, and a visual/typography overhaul - explicitly NOT a written methodology section in the UI (user will cover that in the report/email to the professor instead) and NOT the LLM chatbot (analyzed on request, deliberately not built - see that turn's analysis: high feasibility, ~131K-token context on current Groq models, but real constraint is the same tight OTPM/TPM quota this project has already hit; recommended a separate Groq key if built later, and building it after the professor review rather than before).

Researched current (Sept 2026) "AI slop" UI guidance before touching anything - key takeaways applied: Inter font is now a recognized "default AI tool" tell (avoided - used Source Serif 4 for headings + IBM Plex Sans for body instead), no purple/gradient decoration, no uniform card-radius grids, minimal/no decorative emoji, semantic (not decorative) color use. Sources: [smoothui.dev/blog/ai-design-slop](https://smoothui.dev/blog/ai-design-slop), [925studios.co/blog/ai-slop-web-design-guide](https://www.925studios.co/blog/ai-slop-web-design-guide), [UXPin dashboard design principles](https://www.uxpin.com/studio/blog/dashboard-design-principles/).

Concretely, in `app.py`:
- One CSS block (Source Serif 4 / IBM Plex Sans via Google Fonts, hides Streamlit's default hamburger menu + "Made with Streamlit" footer so the page reads as a finished product).
- `ACTOR_COLORS` - one fixed muted color per actor, applied to the Timeline chart's lines so an actor's color is stable regardless of which actors are toggled on/off. Left the Actor Network's node coloring alone (it already encodes something meaningful - relation-involvement magnitude via a blue gradient - forcing per-actor color there would have destroyed that encoding for no gain).
- `RELATION_TYPE_COLORS` desaturated slightly to match the new palette (same hostile=red/supportive=green semantics, just muted, not the plotting-library-default saturated tab10 colors).
- New at-a-glance summary strip (topic-relevant post count, collection window, most-mentioned actor, most-positive-sentiment actor) right under the header, before the tabs - plain `st.metric` row, no boxed/shadowed cards.
- New computed correlation statistics (both via pandas' own `.corr()`, no new dependency): Timeline tab now shows Pearson r between daily X-mention volume and daily GDELT article volume on overlapping dates; Sentiment tab now shows Pearson r between sentiment_score and (likes+retweets) at the individual-post level. Both are silently omitted (not shown with a caveat paragraph) if there's too little overlapping/varying data to mean anything - never a crash, never a wall of explanation.
- Every long inline error/warning block was cut down hard - e.g. the "data hasn't finished processing" `st.error` was a 6-sentence walkthrough of which scripts to re-run in what order; now a single calm `st.info` line. All decorative emoji (⚠️, 🔄, 🟣) removed. Longer explanatory content that's genuinely useful (chart-reading notes, the betweenness-centrality finding) was kept but moved into `help=` tooltips (native Streamlit hover tooltips, confirmed supported - checked `inspect.signature` against the actually-installed Streamlit 1.62.0) instead of sitting in the page permanently.
- The "Pakistan mediating Iran-US" and "Iran pattern" analytical findings were kept (these are real results, not defensive notes) but restyled into one consistent `.finding-callout` CSS class instead of one-off inline-styled divs, and had their emoji removed.

### Testing before handoff
No new imports were added (correlation uses pandas' built-in `.corr()`), so the dependency-check protocol's "new imports" trigger doesn't strictly apply - still verified beyond `py_compile`: built a synthetic dataset matching every file's real schema (`x_data_cleaned.csv`, `actor_network_v2.json`, `actor_centrality.csv`, `daily_actor_timeline.csv`, `daily_gdelt_timeline.csv`, `hashtag_frequency.csv`, `pipeline_status.log`), actually launched `streamlit run app.py` against it, and drove it with a headless Chromium (Playwright) - screenshotted and text-scanned all 5 tabs for error strings (`Traceback`, `KeyError`, etc.), confirmed the new summary strip/correlation stats/actor colors all render correctly, and separately re-tested with `daily_gdelt_timeline.csv` removed to confirm the GDELT-dependent code (checkbox + correlation) degrades cleanly rather than crashing when that optional file is absent.

### What Claude ran vs. what you need to do
Everything above (the redesign, the `.gitignore` tightening, all testing) was done and verified by Claude and deployed directly to the project folder via the device bridge. Nothing here needs to be run by hand. The still-outstanding manual step from earlier this session - the one-time `git add -A` / commit / push to catch GitHub up - now also carries this UI redesign along with it, since it wasn't pushed separately.

---

---

## 2026-09-10 (later still) — post-review fixes: date bug, node contrast, callout generalization, real theme, git lock, data backlog

User caught several real issues after reviewing the redesigned dashboard locally, before agreeing to push. All were investigated against the actual production `x_data_cleaned.csv` (staged from the device, not guessed at) and fixed in `app.py`; none required touching any script other than `app.py` and the new `.streamlit/config.toml`.

### Bug: impossible-looking "Sept 19 to Sept 9" collection-window date
Root cause, confirmed by loading the real `x_data_cleaned.csv` and inspecting it directly: one real post (id `1968917639519941043`, "Bangladesh should explore... Saudi Arabia-Pakistan Mutual Defense Agreement...") is dated **2025-09-19** - a year before this pact existed - but is correctly `topic_relevant=True` (it's a genuine keyword match about a related defense pact). The summary strip's date-range calculation used the full `topic_relevant==True` set with no lower-date bound, so this one outlier became the computed "earliest" date; since `_fmt_short` deliberately dropped the year (by design, to keep "Aug 7" short), a 2025 date and a 2026 date rendered as "Sep 19 – Sep 9" with nothing distinguishing them - reading as backwards/impossible. The Timeline tab already had a `CHART_MIN_DATE` floor (2026-08-01) built specifically for this class of outlier (see that constant's own comment) - the summary strip just wasn't using it. Fixed by applying the same floor to the summary strip's underlying dataframe, and made `_fmt_short` year-aware (prints the year only if the min/max dates actually span different years) as a defensive second layer. Verified against a synthetic dataset with a planted 2025-09-19 outlier row: window now correctly reads "Aug 8 – Aug 15" instead of including it. The Data Explorer tab intentionally still shows this row (it's real data, not something to hide) - only the summary strip's own stat was wrong.

### Bug: two analytical callouts silently hardcoded to name specific actors
Neither the "Pakistan mediating Iran-US" callout (Actor Network tab) nor the "Iran pattern" callout (Sentiment tab) was actually computed across all 5 actors - both had a specific actor's name baked into the check itself (`_pair_edge("Pakistan", "Iran")`; `report_df[report_df["actor"] == "Iran"]`), so they would keep testing the same named actor forever even if a different actor became the real standout as new data comes in - and structurally, there was never a chance for any actor besides those two to be named, which is what read as arbitrary/incomplete to a reviewer. Rewrote both to compute the relevant statistic (mediating-relation involvement; lowest sentiment + lowest engagement) across all 5 actors uniformly, then name whichever actor the data actually points to. In the current real data this still happens to surface Pakistan and Iran respectively (that's what the underlying relations/sentiment actually show) - the fix is that this is now a genuine computed result, not a coincidence of what the code happened to check for.

### Fix: Actor Network node colors unreadable for Iran/United States
The node fill was plotly's stock "Blues" colorscale (light-to-white at the low end) keyed to each actor's relation-row-involvement value; Iran and United States' values happened to land near the pale end, nearly disappearing against the page's light background. Two changes: (1) each node's outline is now that actor's fixed `ACTOR_COLORS` ring (same colors used on the Timeline chart) at 3px width - this is what actually guarantees visibility regardless of fill lightness, and ties this chart into the same color identity system as the rest of the dashboard; (2) the fill colorscale itself was replaced with a custom two-stop scale (`#a9c9d6` to `#1b4f72`) that never reaches near-white, as a second layer of defense.

### Visual system: from "plain report" to a designed page
Researched current dashboard-design guidance before changing anything - restrained/consistent color use over a "rainbow" palette, visual cohesion via unified styling rather than default-tool flatness, flat/clean charts over 3D effects, clear focus/hover states ([think.design's 2026 dashboard do's/don'ts](https://think.design/blog/dashboard-design-in-2026-dos-and-donts/)) - and Streamlit's own theming system ([Streamlit theming docs](https://docs.streamlit.io/develop/concepts/configuration/theming-customize-colors-and-borders), [config.toml reference](https://docs.streamlit.io/develop/api-reference/configuration/config.toml)), which is the correct way to theme buttons/inputs/focus-states (rather than fighting Streamlit's defaults with CSS alone).

Added `.streamlit/config.toml` (new file, applies automatically both locally and on Streamlit Community Cloud once pushed - no separate dashboard-settings step): a warm paper background (`#faf8f5`, not stark white) with a single deep-teal accent color (`#1f5f5b`) used consistently for buttons, active-tab underline, and focus states, `baseRadius`/`buttonRadius` set to `small` (deliberately not `full` - the uniform-pill-rounded-corner look is one of the specific "AI slop" tells researched earlier this session), and `showWidgetBorder=true` for definition on inputs without resorting to card-shadow grids.

On top of the theme, in `app.py`: every plotly chart now runs through a new `_style_fig()` helper so all 4 charts share the same paper-colored background and warm gridline color instead of each carrying plotly's un-themed white-canvas default; the page title got a colored underline rule (a small "masthead" touch); buttons got a subtle shadow + hover-lift; the active tab gets a colored underline instead of Streamlit's faint default indicator; engagement/hashtag bar charts now use the same teal/amber accent colors instead of plotly's default blue.

One regression caught by the smoke test and fixed before shipping: applying `title_font` unconditionally inside `_style_fig` made plotly.js render a literal bold "undefined" string above the Actor Network chart (which has no title) - `_style_fig` now only sets `title_font` when the figure actually has title text.

### Not done: git `index.lock` failure
User hit `fatal: Unable to create '.../.git/index.lock': File exists` when attempting the one-time catch-up commit - a stale lock file left behind by an interrupted or overlapping git process (a second terminal, or VS Code's own Source Control panel touching the same repo), not a code or pipeline issue. Fix is procedural, not a file change: close any other program that might be touching this repo (VS Code's Source Control tab, GitHub Desktop, another terminal), delete the stale lock file, retry. Exact command given directly to the user (not run by Claude - `device_bash` is still unavailable this session, and this needs to run against the user's actual working tree/terminal session, not a file write).

### Not done: clearing the sentiment-processing backlog
User asked why only 1422 of 2104 English rows show as "fully processed." Checked the real data directly: the 682 "incomplete" rows break down as 255 not yet classified by `relevance_and_relations.py` (`topic_relevant` still blank) + 235 already confirmed `topic_relevant=True` but not yet scored by `add_sentiment.py` + 192 correctly, intentionally skipped (`topic_relevant=False` - off-topic rows are never given a sentiment score by design, that's not a backlog, it's working as intended). So the real backlog is 255+235=490 rows, not 682.

Root cause of why re-running the pipeline hadn't been clearing it: `relevance_and_relations.py`'s default `--sample-size` (400) targets a TOTAL "done" row count, and there are already 1849 done rows (`relations_checked==True`) - meaning `additional_needed = max(0, 400 - 1849) = 0`, so a bare re-run (or one invoked with the default sampling) does nothing at all, silently. `run_pipeline.py` itself already correctly calls this script with `--sample-size 0` (verified by reading it directly - this was fixed in an earlier session, no change needed there), which disables sampling and processes every remaining row - so the automated pipeline is not the problem. The actual cause of the 255-row backlog is the same Groq quota-driven timeout diagnosed earlier this session (`RELATIONS_TIMEOUT_SEC = 2700`, i.e. 45 minutes, imposed by `run_pipeline.py`'s own subprocess wrapper - not a Groq-side limit): a single scheduled run doesn't always finish the full backlog within that window, and checkpointing means it just needs to run again (or run standalone, unbounded) to keep making progress. Commands to clear it fully in one sitting were given directly to the user (not run by Claude - this calls the live Groq API and can take significant wall-clock time; not something to run unattended inside this session).

---

## Standing project instructions (apply to every session, don't lose these)
- Always state plainly whether a script/change was run by Claude directly, or
  needs to be run manually by the user (and why, if it's the latter).
- Before handing over any script with new imports: grep the imports, cross-
  check `requirements.txt`, verify via a fresh venv / `py_compile` before
  calling it done.
- This project's real production files live at
  `C:\Users\zahra\OneDrive\Documents\Project Files` on the user's Windows
  machine, reached via the device bridge. `accounts.db` there is a live
  SQLite file — writing to it through the device bridge has previously hit
  stuck-journal issues (OneDrive sync layer); prefer having the user run
  scripts that touch `accounts.db` locally themselves when in doubt.

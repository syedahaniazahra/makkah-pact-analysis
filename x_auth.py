"""
x_auth.py

Shared cookie-loading + client construction for the X/Twitter fetch layer.
Used by setup_x_account.py, fetch_x_test.py, and fetch_x_data.py.

2026-09-10: the fetch layer was switched from twscrape to twifork (a
maintained fork of twikit; installs as the `twikit` package - see
requirements.txt), after confirming twscrape's cookie-auth path is broken
by a currently-open, unresolved upstream bug: X (via Cloudflare) serving a
challenge-platform bundle instead of the real app to authenticated
requests, which breaks twscrape's anti-bot header computation with either
`XClIdParseError` or `XClIdAccountError` regardless of which cookies are
used. See https://github.com/vladkens/twscrape/issues/330 and
CLAUDE_INVESTIGATION_LOG.md in this folder for the full investigation.
Confirmed via a from-scratch differential test (a brand-new X account,
first-ever cookie use, tested immediately) that this was not specific to
our original account - ruling out "flagged account" and "not enough
cookies" as the cause.

twifork's own is_logged_in() implementation explicitly handles the
Cloudflare-interstitial case that broke twscrape (a 200 response that
isn't JSON gets treated as "not logged in" instead of crashing) - see its
source, twikit/client/client.py - which is why it was chosen over plain
twikit. It uses the same cookie-based auth model we already have
credentials for, so no new account signup or credential-gathering was
needed to switch.

Credentials are read ONLY from a local .env file in this same folder -
never hardcode them here and never paste them into chat. Same two ways to
provide them as before:

  RECOMMENDED: X_COOKIES=...  the ENTIRE "Cookie:" request header value
      from a real, freshly logged-in browser request to x.com.
  Fallback:     X_AUTH_TOKEN=...  and  X_CT0=...  (just those two cookies).

One real format difference from twscrape: twikit's set_cookies() wants a
dict of {name: value}, not a raw "name=value; name2=value2" string -
parse_cookie_string() below handles that conversion; nothing changes about
what you put in .env.

2026-09-10 (later same day) - session persistence + retry, added after a
real Task Scheduler failure (fetch_x_data.py: "NOT logged in", full
writeup in CLAUDE_INVESTIGATION_LOG.md). Two things worth knowing before
touching this file again:

1. Programmatic re-login (username/password) is NOT an option and never
   will be with this library, so don't reach for it as "the real fix."
   twikit's own Client.login() carries this as an internal constant
   (_LOGIN_RETIRED, in twikit/client/client.py): X retired the
   LoginFlow onboarding task this method drove; the real login page now
   posts to an endpoint that requires a ~5 KB $castle_token produced by
   obfuscated in-page JavaScript, with passkey/WebAuthn offered as a
   first factor - none of that is reachable from a plain HTTP client.
   Cookies copied from a real browser session are the only auth path
   available to twikit/twifork (or any similar tool) against X as it
   currently stands. Some periodic manual cookie refresh is therefore
   unavoidable - what's addressable is making that refresh less frequent
   and less confusing when it's needed, which is what the two additions
   below do.

2. What actually changed:
   - COOKIE_JAR persistence (save_session_cookies() / build_client()'s
     jar-loading branch below): X rotates some cookies (ct0 in
     particular) during normal use. Before this, every run re-loaded the
     same static .env snapshot, so a rotated cookie from a previous run
     was silently discarded instead of carried forward - shortening how
     long a single manual .env paste stays useful. Now, right after a
     confirmed-good is_logged_in(), the client's CURRENT cookies (which
     may include values X rotated since .env was last hand-edited) are
     saved to COOKIE_JAR_PATH and used as the starting point for the
     NEXT run, instead of going back to the stale .env snapshot every
     time. The jar is tagged with a fingerprint of the .env cookies it
     was built from, so pasting a genuinely fresh cookie into .env
     always takes over immediately - see _env_cookie_fingerprint().
   - Retry-before-giving-up (is_logged_in_with_retry() below): a single
     is_logged_in()==False isn't proof the session is dead - it's the
     same signal that fires on the known, transient Cloudflare-
     challenge-shell issue (see this module's docstring above and
     CLAUDE_INVESTIGATION_LOG.md), where X serves the anonymous
     interstitial to ONE otherwise-fine request. Before this, a single
     False ended the run immediately, indistinguishable from a truly
     dead session. Now callers that care (setup_x_account.py,
     fetch_x_data.py) retry once after a short pause before concluding
     the session is actually dead and telling the user to refresh
     cookies.
"""

import asyncio
import hashlib
import json
import os

from dotenv import load_dotenv
from twikit import Client

# impersonate value passed to twikit's Client - routes requests through
# curl_cffi's Chrome TLS fingerprint instead of plain httpx. Requires the
# twifork[impersonate] extra (pulls in curl_cffi) - see requirements.txt.
IMPERSONATE = "chrome124"

# Where the persisted session cookie jar lives - see save_session_cookies()/
# build_client() below. COOKIE_JAR_META_PATH just holds a fingerprint of the
# .env cookies the jar was built from, so a fresh manual .env paste is
# always detected and takes priority over a stale jar (see
# _env_cookie_fingerprint()). Neither file is required to exist - both are
# created on first successful login and safely ignored/rebuilt if missing,
# corrupt, or stale.
COOKIE_JAR_PATH = "x_session_cookies.json"
COOKIE_JAR_META_PATH = "x_session_cookies_meta.json"

# How many extra is_logged_in() checks to make (after the first) before
# concluding a session is genuinely dead - see is_logged_in_with_retry().
LOGIN_CHECK_RETRIES = 1
LOGIN_CHECK_RETRY_DELAY_SEC = 8


def parse_cookie_string(raw):
    """Turns a raw "name=value; name2=value2; ..." Cookie header string
    (what X_COOKIES holds) into a {name: value} dict."""
    cookies = {}
    for part in raw.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        name, _, value = part.partition("=")
        name = name.strip()
        value = value.strip()
        if name:
            cookies[name] = value
    return cookies


def debug_print_env_status(raw_values):
    """Print presence/shape of the .env values WITHOUT ever printing the
    full secret - just length and first 4 characters."""
    print("Checking .env values:")
    for var, val in raw_values.items():
        if not val:
            print(f"  {var}: NOT FOUND (0 characters)")
        else:
            print(f"  {var}: {len(val)} characters found (starts with '{val[:4]}')")
    print()


def load_cookie_dict():
    """Same .env precedence as before the twifork switch: X_COOKIES (the
    full cookie string) first, falling back to X_AUTH_TOKEN/X_CT0 alone."""
    load_dotenv()

    full_cookies = os.getenv("X_COOKIES")
    debug_print_env_status({"X_COOKIES": full_cookies})
    if full_cookies:
        cookies = parse_cookie_string(full_cookies)
        print(f"Using X_COOKIES (the full cookie string) - {len(cookies)} cookie(s) found in it.\n")
        return cookies

    print("X_COOKIES not set - falling back to X_AUTH_TOKEN/X_CT0 only.\n")
    raw_values = {"X_AUTH_TOKEN": os.getenv("X_AUTH_TOKEN"), "X_CT0": os.getenv("X_CT0")}
    debug_print_env_status(raw_values)

    missing = [k for k, v in raw_values.items() if not v]
    if missing:
        raise SystemExit(
            f"Missing from .env: {', '.join(missing)}. Set either X_COOKIES "
            "(the full browser Cookie: header - recommended) or both "
            "X_AUTH_TOKEN and X_CT0 in a .env file in this same folder and "
            "re-run."
        )
    return {"auth_token": raw_values["X_AUTH_TOKEN"], "ct0": raw_values["X_CT0"]}


def _env_cookie_fingerprint(cookies):
    """Short, stable hash of whatever cookie dict .env currently resolves
    to (via load_cookie_dict()). Used to tell "the persisted jar still
    matches the credentials in .env" apart from "the user pasted a fresh
    cookie into .env since the jar was last saved" - in the second case the
    jar is from an old session and must be ignored even though it's newer
    on disk. sort_keys=True so the fingerprint doesn't depend on dict
    iteration order (load_cookie_dict()'s two code paths build the dict
    differently)."""
    raw = json.dumps(cookies, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def save_session_cookies(client):
    """Best-effort: snapshot the client's CURRENT cookies (which may
    include values X rotated mid-session, e.g. ct0) to COOKIE_JAR_PATH,
    tagged with a fingerprint of the .env cookies used to start this
    session (see _env_cookie_fingerprint()). Call this right after a
    confirmed-good login check (and again at the end of a long run, if the
    caller wants extra protection against rotation happening mid-run).

    Deliberately never raises - this is a freshness optimization on top of
    a pipeline step that has already done its real work by the time this
    is called (or is about to start real work right after a login check);
    a disk/permission hiccup here should be a printed note, not a reason
    to fail the whole step."""
    try:
        client.save_cookies(COOKIE_JAR_PATH)
        with open(COOKIE_JAR_META_PATH, "w", encoding="utf-8") as f:
            json.dump({"env_fingerprint": _env_cookie_fingerprint(load_cookie_dict())}, f)
    except Exception as exc:
        print(
            f"Note: could not persist session cookies to {COOKIE_JAR_PATH} ({exc}) - "
            f"not fatal, this run's own work is unaffected; the next run will just "
            f"fall back to .env cookies as it always did before this feature existed."
        )


def build_client():
    """Constructs a twikit Client. Prefers a persisted, previously-saved
    session (COOKIE_JAR_PATH) over the static .env snapshot IF the jar was
    built from the same .env cookies still present now - see
    save_session_cookies()/_env_cookie_fingerprint(). This means a cookie
    X rotated (e.g. ct0) during a past successful run carries forward into
    this one, instead of every run discarding that and going back to
    whatever was last hand-pasted into .env. Falls back to .env cookies
    (the original, pre-persistence behavior) whenever the jar is missing,
    unreadable, or stale relative to .env - including the very next run
    after you manually paste a fresh cookie into .env, which always wins
    immediately.

    Does NOT verify the session is actually logged in - callers that care
    should await client.is_logged_in() (or, better, is_logged_in_with_retry()
    below) themselves."""
    env_cookies = load_cookie_dict()
    client = Client("en-US", impersonate=IMPERSONATE)

    if os.path.exists(COOKIE_JAR_PATH) and os.path.exists(COOKIE_JAR_META_PATH):
        try:
            with open(COOKIE_JAR_META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("env_fingerprint") == _env_cookie_fingerprint(env_cookies):
                client.load_cookies(COOKIE_JAR_PATH)
                print(
                    f"Using persisted session cookies from {COOKIE_JAR_PATH} (saved after a "
                    f"previous successful login - may include cookies X rotated since your "
                    f"last manual .env update, e.g. ct0)."
                )
                return client
            print(
                f"{COOKIE_JAR_PATH} exists but the .env cookies have changed since it was "
                f"saved (looks like a fresh manual paste) - using the .env cookies instead."
            )
        except Exception as exc:
            print(
                f"Note: could not read persisted cookies from {COOKIE_JAR_PATH} ({exc}) - "
                f"falling back to .env cookies."
            )

    client.set_cookies(env_cookies)
    return client


async def is_logged_in_with_retry(client, retries=LOGIN_CHECK_RETRIES,
                                   delay_sec=LOGIN_CHECK_RETRY_DELAY_SEC):
    """Wraps client.is_logged_in() with `retries` extra attempts (default
    1) before reporting the session as dead. is_logged_in()==False doesn't
    only mean "these cookies are invalid" - it's the exact same signal
    twifork uses for the known, transient Cloudflare-challenge-shell issue
    (see this module's docstring and CLAUDE_INVESTIGATION_LOG.md): X can
    serve the anonymous interstitial to ONE otherwise-fine request. Without
    this, a single blip looked identical to a genuinely dead session and
    sent you off to re-copy cookies for no reason. Returns the same
    True/False client.is_logged_in() would - callers don't need to know a
    retry happened unless they read the printed notes."""
    ok = await client.is_logged_in()
    if ok or retries <= 0:
        return ok
    for attempt in range(1, retries + 1):
        print(
            f"is_logged_in() returned False (attempt {attempt}/{retries + 1}) - this can be "
            f"a transient Cloudflare-challenge blip (see CLAUDE_INVESTIGATION_LOG.md), not "
            f"necessarily a dead session. Waiting {delay_sec}s and checking again before "
            f"giving up..."
        )
        await asyncio.sleep(delay_sec)
        ok = await client.is_logged_in()
        if ok:
            print(f"Logged in on retry {attempt} - that earlier False was transient.")
            return True
    return False

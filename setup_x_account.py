"""
setup_x_account.py

Phase 3 - verifies the X/Twitter cookies in .env are valid and the session
is genuinely logged in, before fetch_x_test.py / fetch_x_data.py try to use
them.

2026-09-10: rewritten for the twscrape -> twifork/twikit pivot. See
x_auth.py's docstring and CLAUDE_INVESTIGATION_LOG.md in this folder for
why. The old version maintained a local multi-account pool in accounts.db
(a twscrape concept, via api.pool.add_account_cookies()); twikit's Client
works directly off cookies each run instead, so there's no separate
"account pool" file to add/refresh anymore. accounts.db is no longer read
or written by these scripts - safe to ignore, not deleted automatically.

How to get X_COOKIES (recommended) - via DevTools Network tab, no browser
extension needed:
  1. Log into X in a browser with your dedicated scraping account.
  2. Open DevTools (F12) -> Network tab, then reload x.com so a fresh
     request appears in the list.
  3. Click any request to x.com -> Headers tab -> find "Cookie:" under
     Request Headers.
  4. Copy the ENTIRE value after "Cookie: " and paste it as X_COOKIES=...
     in .env (one line, no extra quotes).

How to get X_AUTH_TOKEN / X_CT0 (the older, narrower fallback): DevTools ->
Application/Storage -> Cookies for x.com, and copy the "auth_token" and
"ct0" values individually into .env.

2026-09-10 (later same day): now retries once (via
x_auth.is_logged_in_with_retry()) before reporting a dead session - a
single False can be a transient Cloudflare-challenge blip, not proof the
cookies are actually bad. On a confirmed-good login, also persists the
session's cookies (x_auth.save_session_cookies()) so fetch_x_data.py's
next run starts from this fresh state - including anything X rotated
(e.g. ct0) - instead of only ever re-reading the static .env snapshot.
Programmatic username/password re-login is NOT an option here and never
will be - see x_auth.py's module docstring for why (X retired that flow
entirely); cookies copied from a real browser are the only way in, which
is why this script's whole job is verifying + persisting them as well as
possible, not eliminating the manual step altogether.

Run:
    python setup_x_account.py
"""

import asyncio

from x_auth import build_client, is_logged_in_with_retry, save_session_cookies


async def main():
    print("Building a client from your .env cookies ...\n")
    client = build_client()

    print("Verifying the session is actually logged in on X's side (not "
          "just that cookies are present - twikit calls a real X endpoint "
          "and checks the response) ...")
    try:
        ok = await is_logged_in_with_retry(client)
    except Exception as exc:
        print(f"\nCould not check login status: {exc}")
        print("Common causes: no internet access from wherever this is "
              "running, or X is temporarily unreachable. This is different "
              "from 'not logged in' - that case is reported as False, not "
              "an exception.")
        raise SystemExit(1)

    if ok:
        save_session_cookies(client)
        print("\nLogged in successfully. Next: run fetch_x_test.py to confirm scraping works.")
    else:
        print(
            "\nNOT logged in - X did not accept this session (checked twice, "
            "so this isn't just a one-off blip). Common causes: the cookies "
            "are expired or malformed (re-copy them from a fresh browser "
            "login), or the account is locked/suspended. If you just copied "
            "fresh cookies and this still fails, this can also be the known "
            "Cloudflare-challenge issue described in "
            "CLAUDE_INVESTIGATION_LOG.md rather than a cookie problem - "
            "don't assume it's your credentials without checking there "
            "first."
        )
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())

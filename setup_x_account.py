"""
setup_x_account.py

Phase 3 - adds your dedicated X/Twitter scraping account to twscrape's
local account pool (stored in accounts.db in this folder) using
COOKIE-based authentication, so fetch_x_test.py - and later, the full
collection script - can use it to search tweets.

Why cookies instead of username/password: twscrape's own docs note
username/password login is unreliable and can hit Cloudflare challenges;
cookie-based accounts are the recommended, stable method, and they
activate immediately - no separate login step needed.

Credentials are read ONLY from a local .env file in this same folder -
never hardcode them here and never paste them into chat. Expected keys
in .env:
    X_AUTH_TOKEN=...    (the auth_token cookie value from a logged-in browser session)
    X_CT0=...           (the ct0 cookie value from the same session)

How to get X_AUTH_TOKEN / X_CT0: log into X in a browser with your
dedicated scraping account, open DevTools -> Application/Storage ->
Cookies for x.com, and copy the "auth_token" and "ct0" values into .env.

Account naming: the pool label below (ACCOUNT_LABEL) is a fixed constant,
not read from .env. Earlier runs used a slightly different username
string each time (typos like "Zarmishhx9ua" vs "Zarmishx9ua"), and since
twscrape keys accounts by that string, every typo created a NEW duplicate
row instead of updating the existing one. Using one hardcoded constant
here means every run refers to the same account - if you want to rename
it, edit ACCOUNT_LABEL below (in one place) rather than a .env value.

Run:
    python setup_x_account.py
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from twscrape import API

REQUIRED_VARS = ["X_AUTH_TOKEN", "X_CT0"]
ACCOUNT_LABEL = "makkah_pact_x_scraper"  # fixed on purpose - see note above


def debug_print_env_status(raw_values):
    """Print presence/shape of the .env values WITHOUT ever printing the
    full secret - just length and first 4 characters, so we can tell
    ".env isn't being read" apart from "cookies are wrong/expired"."""
    print("Checking .env values:")
    for var, val in raw_values.items():
        if not val:
            print(f"  {var}: NOT FOUND (0 characters)")
        else:
            print(f"  {var}: {len(val)} characters found (starts with '{val[:4]}')")
    print()


def load_credentials():
    load_dotenv()
    raw_values = {var: os.getenv(var) for var in REQUIRED_VARS}
    debug_print_env_status(raw_values)

    missing = [k for k, v in raw_values.items() if not v]
    if missing:
        print(f"Missing from .env: {', '.join(missing)}")
        print("Add these keys to a .env file in this same folder and re-run:")
        for var in REQUIRED_VARS:
            print(f"  {var}=...")
        sys.exit(1)
    return raw_values


async def main():
    creds = load_credentials()
    api = API()  # uses accounts.db in the current folder

    cookies = f"auth_token={creds['X_AUTH_TOKEN']}; ct0={creds['X_CT0']}"

    print(f"Adding account '{ACCOUNT_LABEL}' to the pool via cookies ...")
    try:
        # Cookie-based accounts activate immediately - unlike
        # username/password accounts, there is NO login_all() step here.
        await api.pool.add_account(
            ACCOUNT_LABEL,
            "",  # password - not used for cookie auth
            "",  # email - not used for cookie auth
            "",  # email_password - not used for cookie auth
            cookies=cookies,
        )
    except Exception as exc:
        print(f"\nCould not add the account: {exc}")
        print("Common causes: no internet access from wherever this is "
              "running, or X_AUTH_TOKEN/X_CT0 in .env are missing/expired.")
        sys.exit(1)

    try:
        accounts = await api.pool.accounts_info()
        print("\nAccount pool status:")
        for acc in accounts:
            print(
                f"  {acc.get('username')}: "
                f"logged_in={acc.get('logged_in')} "
                f"active={acc.get('active')} "
                f"error={acc.get('error_msg')}"
            )
        active_ok = any(acc.get("active") for acc in accounts)
    except Exception as exc:
        print(f"(Could not read detailed pool status: {exc})")
        print("That's not necessarily fatal - fetch_x_test.py will confirm "
              "the account works by actually running a search.")
        active_ok = None

    if active_ok is False:
        print("\nAccount was added but is not showing active - "
              "X_AUTH_TOKEN/X_CT0 in .env are likely missing, malformed, or "
              "expired (cookies expire - you may need to re-copy them from "
              "a fresh browser session).")
        sys.exit(1)

    print("\nDone. Next: run fetch_x_test.py to confirm scraping works.")


if __name__ == "__main__":
    asyncio.run(main())

"""
check_dependencies.py

Run this BEFORE `streamlit run app.py` (or any other script in this
project) whenever you've just pulled updated files. It checks that every
package listed in requirements.txt is actually importable in the CURRENT
Python interpreter - the same interpreter that will run this script.

Why this matters: "pip install -r requirements.txt" and "streamlit run
app.py" only agree with each other if they're run with the same Python.
On Windows especially, it's easy to have more than one Python on your
machine (e.g. one from the Microsoft Store, one from python.org, one
inside a virtualenv) - installing a package with one and running your
script with another is the single most common cause of a
"ModuleNotFoundError" for a package you're SURE you installed.

This script parses requirements.txt directly (rather than hardcoding a
package list), so it stays correct as packages are added/removed - no
separate list to keep in sync.

Run:
    python check_dependencies.py

If anything is missing, it prints the exact command to fix it, using
THIS interpreter (sys.executable), so there's no ambiguity about which
pip/python to use.

IMPORTANT - this checks the interpreter running THIS script, not
necessarily the one that runs when you type `streamlit run app.py`. On
Windows, the bare `streamlit` command is a launcher .exe on your PATH,
and if you have more than one Python installed, that launcher can belong
to a COMPLETELY DIFFERENT Python than the `python` command you just used
here - even though both are "just streamlit" to you. This script always
prints the guaranteed-safe way to launch the app at the end, specifically
so you don't hit that gap: `python -m streamlit run app.py` (using the
exact python.exe this check just ran with) instead of a bare
`streamlit run app.py`.
"""

import importlib
import re
import sys
from pathlib import Path

REQUIREMENTS_FILE = Path(__file__).parent / "requirements.txt"

# A handful of packages have a pip/PyPI name that differs from the name
# you actually `import` in Python. Everything not listed here is assumed
# to import under its own pip name (true for most packages, including
# streamlit, plotly, networkx, numpy, pandas, requests, twscrape,
# rapidfuzz, langdetect, better_profanity, nltk, vaderSentiment).
PIP_NAME_TO_IMPORT_NAME = {
    "python-dotenv": "dotenv",
}


def parse_requirements(path):
    if not path.exists():
        print(f"Could not find {path}")
        sys.exit(1)
    names = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Strip a version specifier like ">=1.24", "==3.0", etc.
        base = re.split(r"[<>=!~\[]", line, maxsplit=1)[0].strip()
        if base:
            names.append(base)
    return names


def main():
    print(f"Checking dependencies with: {sys.executable}\n")

    pip_names = parse_requirements(REQUIREMENTS_FILE)
    missing = []

    for pip_name in pip_names:
        import_name = PIP_NAME_TO_IMPORT_NAME.get(pip_name, pip_name)
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "")
            print(f"  OK    {pip_name:20s} (import {import_name}) {version}")
        except ImportError as exc:
            print(f"  MISSING  {pip_name:20s} (import {import_name}) -> {exc}")
            missing.append(pip_name)

    print()
    if missing:
        print(f"{len(missing)} package(s) missing from this interpreter: {missing}")
        print("\nFix by running (with THIS SAME python/pip):")
        print(f'    "{sys.executable}" -m pip install -r "{REQUIREMENTS_FILE}"')
        print(
            "\nIf you already ran pip install and still see this, you likely have "
            "more than one Python installed and pip installed to a different one "
            "than the command above points to - use the exact command printed "
            "above (it's pinned to this interpreter) rather than a bare `pip "
            "install ...`."
        )
    else:
        print("All packages in requirements.txt import successfully with this interpreter.")

    # This matters even when everything above passed: a bare `streamlit` command
    # is a separate launcher .exe on your PATH, and on a machine with more than
    # one Python installed, that launcher can belong to a DIFFERENT Python than
    # the one that just ran this check - so "all OK here" does not guarantee
    # `streamlit run app.py` will actually use this same interpreter. Launching
    # via `-m` pins it to this exact python.exe, closing that gap.
    print(
        "\nTo guarantee `streamlit run` uses THIS SAME interpreter (the bare "
        "`streamlit` command on your PATH may launch a different Python's "
        "install if you have more than one), run:"
    )
    print(f'    "{sys.executable}" -m streamlit run app.py')

    if missing:
        sys.exit(1)


if __name__ == "__main__":
    main()

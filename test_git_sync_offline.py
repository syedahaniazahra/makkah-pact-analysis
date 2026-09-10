"""
test_git_sync_offline.py

Standalone offline test harness for the new PAT-based sync_to_git() in
run_pipeline.py - exercises the REAL function (imported from the actual
file, not reimplemented) against a fake local "GitHub" remote, so nothing
here ever touches the real repo or a real network/GitHub endpoint.

How the fake remote works: a local bare git repo stands in for GitHub.
sync_to_git() only ever knows how to build push URLs shaped like
"https://<PAT>@github.com/<owner>/<repo>.git" (matching real GitHub), so
this test uses `git config url.<bare-repo-path>.insteadOf <that exact
fake-github URL>` - a standard git URL-rewrite mechanism - so that when
the code pushes to what it thinks is github.com, git transparently
redirects the operation to the local bare repo instead. This means the
test exercises the actual push mechanics end-to-end (auth URL
construction, the real `git push` subprocess call, real success/failure
handling) without any code path being mocked out.

Run: python test_git_sync_offline.py
"""

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile

RUN_PIPELINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_pipeline.py")

FAKE_OWNER = "testowner"
FAKE_REPO = "testrepo"
FAKE_TOKEN = "ghp_FAKETESTTOKEN1234567890"
FAKE_ORIGIN_URL = f"https://github.com/{FAKE_OWNER}/{FAKE_REPO}.git"
FAKE_AUTH_URL = f"https://{FAKE_TOKEN}@github.com/{FAKE_OWNER}/{FAKE_REPO}.git"

PASS = []
FAIL = []


def check(name, condition, extra=""):
    if condition:
        PASS.append(name)
        print(f"  [PASS] {name}")
    else:
        FAIL.append(name)
        print(f"  [FAIL] {name} {extra}")


def run(cmd, cwd, check_rc=True, env=None):
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
    if check_rc and proc.returncode != 0:
        raise RuntimeError(f"command failed: {cmd}\nstdout={proc.stdout}\nstderr={proc.stderr}")
    return proc


def load_module():
    spec = importlib.util.spec_from_file_location("run_pipeline_under_test", RUN_PIPELINE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_test_repo(tmp_root):
    """Bare repo (fake GitHub) + a working local repo that mimics the real
    project's layout (origin = a github.com URL, main branch, one existing
    commit already present so 'nothing changed' is a real, testable state)."""
    bare_dir = os.path.join(tmp_root, "fake_origin.git")
    local_dir = os.path.join(tmp_root, "local_repo")
    os.makedirs(bare_dir)
    os.makedirs(local_dir)

    run(["git", "init", "--bare", "-b", "main"], cwd=bare_dir)
    run(["git", "init", "-b", "main"], cwd=local_dir)
    run(["git", "config", "user.name", "Test User"], cwd=local_dir)
    run(["git", "config", "user.email", "test@example.com"], cwd=local_dir)
    run(["git", "remote", "add", "origin", FAKE_ORIGIN_URL], cwd=local_dir)
    # The URL-rewrite: redirect the exact authenticated URL the code will
    # build to the local bare repo instead of real github.com.
    run(["git", "config", f"url.{bare_dir}.insteadOf", FAKE_AUTH_URL], cwd=local_dir)

    # Seed an initial file + commit + push, so the repo starts in a normal
    # "already has history" state, same as the real project.
    with open(os.path.join(local_dir, "test_data.csv"), "w") as f:
        f.write("a,b\n1,2\n")
    run(["git", "add", "test_data.csv"], cwd=local_dir)
    run(["git", "commit", "-m", "initial commit"], cwd=local_dir)
    run(["git", "push", FAKE_AUTH_URL, "HEAD:main"], cwd=local_dir)
    return bare_dir, local_dir


def bare_log(bare_dir):
    proc = run(["git", "log", "--oneline", "main"], cwd=bare_dir, check_rc=False)
    return proc.stdout


def main():
    tmp_root = tempfile.mkdtemp(prefix="git_sync_test_")
    print(f"Test sandbox: {tmp_root}\n")
    try:
        # ---------------------------------------------------------------
        print("Test 1: GITHUB_PAT not set -> clean SKIP, no commit attempted")
        bare_dir, local_dir = make_test_repo(os.path.join(tmp_root, "t1"))
        mod = load_module()
        mod.DATA_DIR = local_dir
        mod.SYNC_FILES = ["test_data.csv"]
        os.environ.pop("GITHUB_PAT", None)
        # Modify the tracked file so there WOULD be something to commit,
        # to prove the missing-PAT check happens before add/commit at all.
        with open(os.path.join(local_dir, "test_data.csv"), "a") as f:
            f.write("3,4\n")
        before_log = bare_log(bare_dir)
        result = mod.sync_to_git([])
        after_log = bare_log(bare_dir)
        check("status is SKIPPED", result["status"] == "SKIPPED", result)
        check("detail mentions GITHUB_PAT", "GITHUB_PAT" in result["detail"], result)
        check("bare repo unchanged (no push happened)", before_log == after_log)
        # revert the uncommitted edit for the next test
        run(["git", "checkout", "--", "test_data.csv"], cwd=local_dir)

        # ---------------------------------------------------------------
        print("\nTest 2: GITHUB_PAT set, nothing changed -> SUCCESS, no-op")
        os.environ["GITHUB_PAT"] = FAKE_TOKEN
        before_log = bare_log(bare_dir)
        result = mod.sync_to_git([])
        after_log = bare_log(bare_dir)
        check("status is SUCCESS", result["status"] == "SUCCESS", result)
        check("detail says no changes", "no changes" in result["detail"], result)
        check("bare repo unchanged", before_log == after_log)

        # ---------------------------------------------------------------
        print("\nTest 3: GITHUB_PAT set, real change -> SUCCESS, actually pushed, no token leak")
        with open(os.path.join(local_dir, "test_data.csv"), "a") as f:
            f.write("5,6\n")
        fake_results = [{"label": "fetch_x_data", "status": "SUCCESS", "detail": "12 new row(s)", "elapsed": 1.0}]
        before_log = bare_log(bare_dir)
        result = mod.sync_to_git(fake_results)
        after_log = bare_log(bare_dir)
        check("status is SUCCESS", result["status"] == "SUCCESS", result)
        check("detail says pushed", "pushed" in result["detail"], result)
        check("bare repo received the new commit", after_log != before_log, f"before={before_log!r} after={after_log!r}")
        check("commit message includes new-row detail", "12 new row(s)" in run(["git", "log", "-1", "--format=%s"], cwd=bare_dir).stdout)
        check("FAKE_TOKEN not present in result detail", FAKE_TOKEN not in result["detail"], result)

        # ---------------------------------------------------------------
        print("\nTest 4: push rejected (remote diverged) -> FAILED gracefully, no crash, no token leak")
        # Simulate someone/something else pushing directly to the bare repo,
        # so our local_dir's next push is rejected as non-fast-forward.
        other_clone = os.path.join(tmp_root, "t1", "other_clone")
        run(["git", "clone", bare_dir, other_clone], cwd=tmp_root)
        run(["git", "config", "user.name", "Other"], cwd=other_clone)
        run(["git", "config", "user.email", "other@example.com"], cwd=other_clone)
        with open(os.path.join(other_clone, "test_data.csv"), "a") as f:
            f.write("9,9\n")
        run(["git", "add", "test_data.csv"], cwd=other_clone)
        run(["git", "commit", "-m", "a divergent commit from elsewhere"], cwd=other_clone)
        run(["git", "push", "origin", "HEAD:main"], cwd=other_clone)

        with open(os.path.join(local_dir, "test_data.csv"), "a") as f:
            f.write("7,8\n")
        result = mod.sync_to_git([])
        check("status is FAILED (not a crash/exception)", result["status"] == "FAILED", result)
        check("FAKE_TOKEN not present in result detail", FAKE_TOKEN not in result["detail"], result)
        check("detail is non-empty and informative", bool(result["detail"]) and len(result["detail"]) > 5, result)

        # ---------------------------------------------------------------
        print("\nTest 5: bad/unrecognized origin URL -> FAILED gracefully, no crash")
        bare_dir5, local_dir5 = make_test_repo(os.path.join(tmp_root, "t5"))
        mod5 = load_module()
        mod5.DATA_DIR = local_dir5
        mod5.SYNC_FILES = ["test_data.csv"]
        run(["git", "remote", "set-url", "origin", "https://gitlab.example.com/owner/repo.git"], cwd=local_dir5)
        with open(os.path.join(local_dir5, "test_data.csv"), "a") as f:
            f.write("1,1\n")
        os.environ["GITHUB_PAT"] = FAKE_TOKEN
        result = mod5.sync_to_git([])
        check("status is FAILED (not a crash)", result["status"] == "FAILED", result)
        check("detail explains the unrecognized remote", "github.com" in result["detail"].lower() or "recognizable" in result["detail"].lower(), result)

    finally:
        os.environ.pop("GITHUB_PAT", None)
        shutil.rmtree(tmp_root, ignore_errors=True)

    print(f"\n{'='*60}")
    print(f"RESULTS: {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED CHECKS:")
        for f in FAIL:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()

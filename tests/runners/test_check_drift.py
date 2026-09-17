#!/usr/bin/env python3
"""
End-to-end test for the --check-drift CLI, run as real subprocesses against
a real notebook file on disk -- not in-process function calls.

test_manifest_roundtrip_e2e.py (pytest, tests/) already covers generation,
extraction, and check-drift by calling the Python functions directly. That
can never verify the actual CLI plumbing itself: argparse parsing real
sys.argv, sys.exit() producing a real OS-level process exit code, the two
invocations (generate, then check) genuinely being separate processes
against the same file on disk. This script exists specifically to prove
that plumbing, which a direct function call structurally cannot.

Invoked directly (matches this directory's other runners, not pytest):
    python tests/runners/test_check_drift_e2e.py
"""
import sys
import json
import subprocess
import os

FIXTURE_PATH = "tests/fixtures/temp_check_drift_fixture.ipynb"
MERGED_PATH = "tests/fixtures/temp_check_drift_fixture_merged.ipynb"
NO_MANIFEST_PATH = "tests/fixtures/temp_check_drift_no_manifest.ipynb"

# requests==2.32.0 is permanently yanked (CVE-2024-35195 mitigation conflict) --
# a durable, verified-earlier fact, safe to assert against real PyPI without
# this test breaking on schedule rather than on regression.
YANKED_PACKAGE = "requests"
YANKED_VERSION = "2.32.0"


def _write_notebook(path, cell_source):
    content = {
        "cells": [{"cell_type": "code", "source": [cell_source], "metadata": {}}],
        "metadata": {}, "nbformat": 4, "nbformat_minor": 5,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(content, f)


def _cleanup():
    for path in (FIXTURE_PATH, MERGED_PATH, NO_MANIFEST_PATH):
        if os.path.exists(path):
            os.remove(path)
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "-q", YANKED_PACKAGE],
        capture_output=True, text=True,
    )


def test_generate_then_check_drift_real_subprocesses() -> int:
    """Real generate, real check-drift, two separate processes, same file."""
    print(f"1. Installing {YANKED_PACKAGE}=={YANKED_VERSION} (a real, permanently-yanked release)...")
    install = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", f"{YANKED_PACKAGE}=={YANKED_VERSION}"],
        capture_output=True, text=True,
    )
    if install.returncode != 0:
        print(f"FAIL: could not install {YANKED_PACKAGE}=={YANKED_VERSION}.\n{install.stderr}")
        return 1

    _write_notebook(FIXTURE_PATH, f"import {YANKED_PACKAGE}")

    print("2. Generating (real subprocess: notebook_env.py <fixture> --output)...")
    gen = subprocess.run(
        [sys.executable, "notebook_env.py", FIXTURE_PATH, "--output"],
        capture_output=True, text=True,
    )
    if gen.returncode != 0:
        print(f"FAIL: generation subprocess exited {gen.returncode}.\n{gen.stderr}\n{gen.stdout}")
        return 1
    if not os.path.exists(MERGED_PATH):
        print(f"FAIL: expected merged output at {MERGED_PATH}, not found.")
        return 1
    print("   PASS: generation succeeded, merged file written.")

    print("3. Checking drift (real subprocess: notebook_env.py <merged> --check-drift)...")
    check = subprocess.run(
        [sys.executable, "notebook_env.py", MERGED_PATH, "--check-drift"],
        capture_output=True, text=True,
    )
    if check.returncode != 1:
        print(f"FAIL: expected real exit code 1 (confirmed drift), got {check.returncode}.\n{check.stdout}\n{check.stderr}")
        return 1
    if "[yanked]" not in check.stdout:
        print(f"FAIL: expected '[yanked]' finding in check-drift output, not found.\n{check.stdout}")
        return 1
    print("   PASS: check-drift correctly exited 1 with a real [yanked] finding.")
    return 0


def test_check_drift_no_manifest_exits_zero() -> int:
    """A real file with no STEADY_PY_MANIFEST -- real subprocess, real exit 0."""
    print("4. Checking drift against a real file with no manifest present...")
    _write_notebook(NO_MANIFEST_PATH, "print('no manifest here')")

    check = subprocess.run(
        [sys.executable, "notebook_env.py", NO_MANIFEST_PATH, "--check-drift"],
        capture_output=True, text=True,
    )
    if check.returncode != 0:
        print(f"FAIL: expected real exit code 0 (nothing to check), got {check.returncode}.\n{check.stdout}\n{check.stderr}")
        return 1
    print("   PASS: check-drift correctly exited 0 with no manifest present.")
    return 0


def test_check_drift_tampering_real_subprocess() -> int:
    """Hand-edit the real merged file on disk after generation, then re-check via
    a fresh subprocess -- proves tampering detection survives an actual
    generate-then-edit-then-check workflow, not just an in-memory round trip."""
    print("5. Hand-editing the merged file on disk, then re-checking...")
    if not os.path.exists(MERGED_PATH):
        print(f"FAIL: {MERGED_PATH} does not exist -- run generation step first.")
        return 1

    with open(MERGED_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    tampered = content.replace(f"'{YANKED_VERSION}'", "'2.32.1'")
    if tampered == content:
        print("FAIL: tampering replacement did not match anything in the merged file.")
        return 1
    with open(MERGED_PATH, "w", encoding="utf-8") as f:
        f.write(tampered)

    check = subprocess.run(
        [sys.executable, "notebook_env.py", MERGED_PATH, "--check-drift"],
        capture_output=True, text=True,
    )
    if check.returncode != 1:
        print(f"FAIL: expected real exit code 1 (tampering is a confirmed finding), got {check.returncode}.\n{check.stdout}")
        return 1
    if "[tampered]" not in check.stdout:
        print(f"FAIL: expected '[tampered]' finding in check-drift output, not found.\n{check.stdout}")
        return 1
    print("   PASS: check-drift correctly detected hand-edited manifest on disk.")
    return 0


def main() -> int:
    try:
        for test_fn in (
            test_generate_then_check_drift_real_subprocesses,
            test_check_drift_no_manifest_exits_zero,
            test_check_drift_tampering_real_subprocess,
        ):
            result = test_fn()
            if result != 0:
                return result
        return 0
    finally:
        _cleanup()


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""End-to-end test for the --check-drift CLI.

Executes real subprocesses against actual notebook files on disk to verify
CLI argument handling, process exit codes, and manifest drift detection.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

from e2e_harness import (
    FIXTURES_DIR,
    fail_test,
    run_cli_command,
    run_notebook_env,
    temp_notebook,
)

FIXTURE_PATH = FIXTURES_DIR / "temp_check_drift_fixture.ipynb"
MERGED_PATH = FIXTURES_DIR / "temp_check_drift_fixture_merged.ipynb"
NO_MANIFEST_PATH = FIXTURES_DIR / "temp_check_drift_no_manifest.ipynb"

YANKED_PACKAGE = "requests"
YANKED_VERSION = "2.32.0"


def cleanup_pip_package() -> None:
    """Removes the yanked package from the test environment."""
    run_cli_command([sys.executable, "-m", "pip", "uninstall", "-y", "-q", YANKED_PACKAGE])


def run_generate_then_check_drift() -> None:
    """Tests generating a manifest and detecting drift with a yanked package."""
    print(f"1. Installing {YANKED_PACKAGE}=={YANKED_VERSION} (yanked release)...")
    install = run_cli_command([sys.executable, "-m", "pip", "install", "--quiet", f"{YANKED_PACKAGE}=={YANKED_VERSION}"])
    if not install.ok:
        fail_test(
            "Install Yanked Package",
            f"Failed to install {YANKED_PACKAGE}=={YANKED_VERSION}",
            stdout=install.stdout,
            stderr=install.stderr,
        )

    with temp_notebook(FIXTURE_PATH, [f"import {YANKED_PACKAGE}"]):
        print("2. Generating (notebook_env.py <fixture> --output)...")
        gen = run_notebook_env(str(FIXTURE_PATH), "--output")
        if not gen.ok:
            fail_test(
                "Generate Manifest Subprocess",
                f"notebook_env exited with status {gen.returncode}",
                stdout=gen.stdout,
                stderr=gen.stderr,
            )

        if not MERGED_PATH.exists():
            fail_test("Merged File Generation", f"Expected output file not found: {MERGED_PATH}")

        try:
            print("   PASS: generation succeeded, merged file written.")

            print("3. Checking drift (notebook_env.py <merged> --check-drift)...")
            check = run_notebook_env(str(MERGED_PATH), "--check-drift")
            if check.returncode != 1:
                fail_test(
                    "Check Drift Status Code",
                    f"Expected exit code 1, received {check.returncode}",
                    stdout=check.stdout,
                    stderr=check.stderr,
                )

            if "[yanked]" not in check.stdout:
                fail_test(
                    "Check Drift Findings",
                    "Expected '[yanked]' finding not present in output.",
                    stdout=check.stdout,
                    stderr=check.stderr,
                )
            print("   PASS: check-drift correctly exited 1 with a real [yanked] finding.")

            print("5. Hand-editing the merged file on disk, then re-checking...")
            content = MERGED_PATH.read_text(encoding="utf-8")
            tampered = content.replace(f"'{YANKED_VERSION}'", "'2.32.1'")
            if tampered == content:
                fail_test("Tampering Setup", "Replacement target not found in generated manifest.")
            MERGED_PATH.write_text(tampered, encoding="utf-8")

            tamper_check = run_notebook_env(str(MERGED_PATH), "--check-drift")
            if tamper_check.returncode != 1:
                fail_test(
                    "Tampering Detection Exit Code",
                    f"Expected exit code 1 for tampered manifest, received {tamper_check.returncode}",
                    stdout=tamper_check.stdout,
                    stderr=tamper_check.stderr,
                )

            if "[tampered]" not in tamper_check.stdout:
                fail_test(
                    "Tampering Detection Findings",
                    "Expected '[tampered]' finding not present in output.",
                    stdout=tamper_check.stdout,
                    stderr=tamper_check.stderr,
                )
            print("   PASS: check-drift correctly detected hand-edited manifest on disk.")
        finally:
            if MERGED_PATH.exists():
                MERGED_PATH.unlink()


def run_check_drift_no_manifest() -> None:
    """Verifies that running check-drift on an unmanaged notebook exits cleanly."""
    print("4. Checking drift against a real file with no manifest present...")
    with temp_notebook(NO_MANIFEST_PATH, ["print('no manifest here')"]):
        check = run_notebook_env(str(NO_MANIFEST_PATH), "--check-drift")
        if not check.ok:
            fail_test(
                "Unmanaged Notebook Check",
                f"Expected exit code 0, received {check.returncode}",
                stdout=check.stdout,
                stderr=check.stderr,
            )
        print("   PASS: check-drift correctly exited 0 with no manifest present.")


def main() -> None:
    try:
        run_generate_then_check_drift()
        run_check_drift_no_manifest()
    finally:
        cleanup_pip_package()


if __name__ == "__main__":
    main()
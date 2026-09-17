#!/usr/bin/env python3
"""Phase 5g: Live-Kernel Stale Module Test.

Generates a pin manifest via notebook_env.py, simulates a user importing a package,
runs the newly inserted setup cell to re-pin it, and confirms that in-memory modules
remain stale without a kernel restart.
"""

from __future__ import annotations

from pathlib import Path
import sys

from e2e_harness import (
    FIXTURES_DIR,
    fail_test,
    get_cell_source,
    interactive_kernel,
    load_notebook,
    run_notebook_env,
    temp_notebook,
)


def main() -> None:
    fixture_path = FIXTURES_DIR / "temp_stale_fixture.ipynb"
    initial_cells = ["!pip install numpy==1.23.5\nimport numpy"]

    with temp_notebook(fixture_path, initial_cells):
        print("1. Running notebook_env.py to generate the setup cell...")
        res = run_notebook_env(str(fixture_path), "--in-place")
        if not res.ok:
            fail_test(
                "Generate Setup Cell",
                f"notebook_env.py returned exit code {res.returncode}",
                stdout=res.stdout,
                stderr=res.stderr,
            )
        print(res.stdout.strip())

        nb_data = load_notebook(fixture_path)
        # Cell 0 is markdown, Cell 1 is the sequential installer
        setup_cell_code = get_cell_source(nb_data, 1)

        print("\nStarting interactive kernel...")
        with interactive_kernel("python3") as kernel:
            print("2. Importing numpy interactively...")
            out1 = kernel.execute("import numpy\nprint(numpy.__version__)")
            if not out1.ok:
                fail_test("Initial Import", "Failed to import numpy", stderr="\n".join(out1.errors))

            initial_version = out1.stdout.strip()
            print(f"   Initial version loaded: {initial_version}")

            print("\n3. Executing generated Cell 2...")
            out2 = kernel.execute(setup_cell_code)
            if not out2.ok:
                fail_test("Execute Setup Cell", "Setup cell execution failed", stderr="\n".join(out2.errors))

            if "you may need to restart the kernel" not in out2.stdout.lower():
                fail_test(
                    "Verify Pip Advisory",
                    "Kernel restart advisory not surfaced in setup cell output.",
                    stdout=out2.stdout,
                )
            print("   PASS: pip kernel-restart advisory is correctly surfaced in output.")

            print("\n4. Checking active module state post-install...")
            out3 = kernel.execute("import numpy\nprint(numpy.__version__)")
            if not out3.ok:
                fail_test("Post-Install Check", "Import execution failed", stderr="\n".join(out3.errors))

            stale_version = out3.stdout.strip()
            print(f"   Version currently active in memory: {stale_version}")

            if stale_version != initial_version:
                fail_test(
                    "Verify Module Staleness",
                    f"Kernel unexpectedly reloaded package in-memory. Expected {initial_version}, got {stale_version}",
                )
            print("   PASS: Module behaves as stale, confirming the live-session risk.")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Phase 0: Live-Kernel Regressions Test.

Simulates an interactive session: executes regular user cells, runs notebook_env.py
directly inside the active kernel, and verifies sys.argv isolation, duplicate log
handler prevention, and cell execution history introspection.
"""

from __future__ import annotations

from pathlib import Path
import sys

from e2e_harness import (
    WORKSPACE_ROOT,
    fail_test,
    interactive_kernel,
)


def main() -> None:
    tool_path = WORKSPACE_ROOT / "notebook_env.py"
    if not tool_path.exists():
        fail_test("Locate Tool Source", f"Could not find {tool_path}")

    tool_source = tool_path.read_text(encoding="utf-8")

    print("Starting interactive kernel...")
    with interactive_kernel("python3") as kernel:
        print("1. Populating kernel history with dummy imports...")
        res1 = kernel.execute("import ipykernel\nprint('Dummy cell executed')")
        if not res1.ok:
            fail_test("Setup Dummy Imports", "Execution failed", stderr="\n".join(res1.errors))

        print("\n2. Executing notebook_env.py (Paste-and-Run 1)...")
        res2 = kernel.execute(tool_source)
        if not res2.ok:
            fail_test(
                "Execute Tool in Kernel",
                "Execution raised errors (probable sys.argv contamination)",
                stderr="\n".join(res2.errors),
                stdout=res2.stdout,
            )
        print("   PASS: sys.argv contamination avoided (tool did not crash).")

        print("\n3. Verifying History Introspection...")
        if "ipykernel" not in res2.stdout.lower():
            preview = res2.stdout[-1000:] if len(res2.stdout) > 1000 else res2.stdout
            fail_test(
                "Inspect Session History",
                "Tool failed to extract prior cell history (ipykernel not found in output).",
                details={"stdout_preview": preview},
            )
        print("   PASS: History introspection correctly captured prior cells.")

        print("\n4. Executing notebook_env.py (Paste-and-Run 2 for Log Handlers)...")
        res3 = kernel.execute(tool_source)
        if not res3.ok:
            fail_test("Repeat Execution in Kernel", "Second execution crashed", stderr="\n".join(res3.errors))

        print("\n5. Checking active log handlers...")
        handler_check = "import logging\nprint(len(logging.getLogger('notebook_env').handlers))"
        res4 = kernel.execute(handler_check)
        if not res4.ok:
            fail_test("Check Log Handlers", "Handler query script failed", stderr="\n".join(res4.errors))

        try:
            handler_count = int(res4.stdout.strip())
        except ValueError:
            fail_test("Parse Log Handlers Output", f"Expected integer count, received: {res4.stdout}")

        if handler_count > 1:
            fail_test(
                "Assert Single Handler",
                f"Duplicate log handlers detected ({handler_count} handlers active).",
            )
        print("   PASS: Duplicate log handlers successfully prevented.")


if __name__ == "__main__":
    main()
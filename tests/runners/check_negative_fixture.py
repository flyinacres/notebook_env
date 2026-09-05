#!/usr/bin/env python3
"""
Structural verification for negative e2e fixtures.

Run against a notebook that was executed with `--allow-errors` (so the
output file always gets written, regardless of cell outcome). Checks that
exactly one cell error occurred, of the expected type and message, rather
than grepping a single traceback string out of terminal stdout/stderr.

Usage:
    python3 check_negative_fixture.py <notebook_path> [expected_ename] [expected_evalue_substring]

Either expected value may be omitted (or passed as an empty string) to skip
that particular check; the exactly-one-error check always runs.

Exit code 0 = structural check passed. Exit code 1 = failed; a message
explaining what was found vs. expected is printed to stdout either way.
"""

import json
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: check_negative_fixture.py <notebook_path> [expected_ename] [expected_evalue_substring]")
        return 1

    notebook_path = sys.argv[1]
    expected_ename = sys.argv[2] if len(sys.argv) > 2 else ""
    expected_evalue_substring = sys.argv[3] if len(sys.argv) > 3 else ""

    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"FAIL: could not read/parse notebook at {notebook_path}: {exc}")
        return 1

    errors = []
    for cell_idx, cell in enumerate(nb.get("cells", [])):
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                errors.append((cell_idx, output.get("ename"), output.get("evalue")))

    if len(errors) != 1:
        print(f"FAIL: expected exactly one cell error, found {len(errors)}")
        for cell_idx, ename, evalue in errors:
            print(f"  cell {cell_idx}: {ename}: {evalue}")
        return 1

    cell_idx, ename, evalue = errors[0]
    ok = True

    if expected_ename and ename != expected_ename:
        print(f"FAIL: expected ename '{expected_ename}' but got '{ename}'")
        ok = False

    if expected_evalue_substring and expected_evalue_substring not in (evalue or ""):
        print(f"FAIL: expected evalue to contain '{expected_evalue_substring}' but got '{evalue}'")
        ok = False

    if ok:
        print(f"PASS: cell {cell_idx} raised {ename}: {evalue}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())

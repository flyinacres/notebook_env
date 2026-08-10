"""
Diagnostic script: recursively scans a directory of .ipynb files and reports
which notebook_env.py code paths each notebook actually exercises.

Not part of notebook_env.py itself -- a one-off tool to answer "does my real
test corpus already cover X" before going out and sourcing new notebooks for
coverage that might already exist.

Usage:
    python diagnose_coverage.py [root_dir]     # default root_dir: test_notebooks
"""

import json
import re
import sys
from pathlib import Path

import notebook_env as ne

SKIP_DIRS = {".git", ".ipynb_checkpoints", "venv", "env", "__pycache__"}

# Cell magics classify_cell_source() doesn't currently recognize at all
# (not SHELL_SCRIPT, not WRITEFILE) -- these fall through to PYTHON and get
# ast.parse'd, which will SyntaxError and be silently dropped.
KNOWN_UNHANDLED_MAGICS = {"%%sql", "%%html", "%%time", "%%timeit", "%%capture", "%%javascript"}

HARDWARE_TAG_PATTERN = re.compile(r'[\w\-]+==[\d.]+\+[\w]+')
RELATIVE_IMPORT_PATTERN = re.compile(r'^\s*from\s+\.+\s*(?:\.\w+)*\s+import\s', re.MULTILINE)


def find_notebooks(root: Path):
    for path in sorted(root.rglob("*.ipynb")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def scan_notebook(path: Path) -> dict:
    findings = {
        "path": str(path),
        "parse_error": None,
        "guarded_imports": set(),
        "dynamic_warnings": [],
        "unhandled_magics": set(),
        "writefile_cells": 0,
        "shell_cells": 0,
        "magic_warnings": [],
        "magic_notices": [],
        "harvested_pkgs": set(),
        "hardware_tags": set(),
        "relative_imports": False,
        "python_version": None,
    }

    success, imports, submodules, code_sources, err, lang_label, guarded, dyn_warns = ne.extract_from_file(str(path))
    if not success:
        findings["parse_error"] = err
        return findings

    findings["guarded_imports"] = guarded
    findings["dynamic_warnings"] = dyn_warns

    try:
        nb_data = json.loads(path.read_text(encoding="utf-8"))
        py_ver = nb_data.get("metadata", {}).get("language_info", {}).get("version")
        findings["python_version"] = py_ver
    except Exception:
        pass

    for source in code_sources:
        cell_type, _ = ne.classify_cell_source(source)
        if cell_type == "WRITEFILE":
            findings["writefile_cells"] += 1
        elif cell_type == "SHELL_SCRIPT":
            findings["shell_cells"] += 1

        first_line = source.splitlines()[0].strip() if source.splitlines() else ""
        first_token = first_line.split()[0] if first_line.split() else ""
        if first_token in KNOWN_UNHANDLED_MAGICS:
            findings["unhandled_magics"].add(first_token)

        if HARDWARE_TAG_PATTERN.search(source):
            for m in HARDWARE_TAG_PATTERN.findall(source):
                findings["hardware_tags"].add(m)

        if RELATIVE_IMPORT_PATTERN.search(source):
            findings["relative_imports"] = True

    harvested_pkgs, base_urls, extra_urls, magic_warnings, magic_notices = ne.harvest_cell_magics_and_commands(code_sources)
    findings["harvested_pkgs"] = harvested_pkgs
    findings["magic_warnings"] = magic_warnings
    findings["magic_notices"] = magic_notices

    return findings


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("test_notebooks")
    if not root.exists():
        print(f"Directory not found: {root}")
        sys.exit(1)

    notebooks = list(find_notebooks(root))
    print(f"Scanning {len(notebooks)} notebooks under {root}\n")

    all_findings = [scan_notebook(p) for p in notebooks]

    # --- Aggregate counts ---
    with_guarded = [f for f in all_findings if f["guarded_imports"]]
    with_dyn_warn = [f for f in all_findings if f["dynamic_warnings"]]
    with_writefile = [f for f in all_findings if f["writefile_cells"] > 0]
    with_shell = [f for f in all_findings if f["shell_cells"] > 0]
    with_unhandled_magic = [f for f in all_findings if f["unhandled_magics"]]
    with_magic_warn = [f for f in all_findings if f["magic_warnings"]]
    with_magic_notice = [f for f in all_findings if f["magic_notices"]]
    with_hw_tags = [f for f in all_findings if f["hardware_tags"]]
    with_rel_import = [f for f in all_findings if f["relative_imports"]]
    with_parse_error = [f for f in all_findings if f["parse_error"]]
    py2_notebooks = [f for f in all_findings if f["python_version"] and f["python_version"].startswith("2.")]

    print("=" * 70)
    print("COVERAGE SUMMARY")
    print("=" * 70)
    print(f"Total notebooks scanned:              {len(all_findings)}")
    print(f"Parse errors:                         {len(with_parse_error)}")
    print(f"Guarded imports (try/except):          {len(with_guarded)}")
    print(f"Dynamic import warnings:               {len(with_dyn_warn)}")
    print(f"%%writefile cells:                     {len(with_writefile)}")
    print(f"%%bash/%%sh/etc shell cells:            {len(with_shell)}")
    print(f"Unhandled cell magics (%%sql etc):      {len(with_unhandled_magic)}")
    print(f"Magic warnings (e.g. -r requirements):  {len(with_magic_warn)}")
    print(f"Magic notices (conda/apt-get/yum):      {len(with_magic_notice)}")
    print(f"Hardcoded hardware-tagged versions:     {len(with_hw_tags)}")
    print(f"Bare relative imports (from . import):  {len(with_rel_import)}")
    print(f"Python 2.x notebooks:                   {len(py2_notebooks)}")
    print()

    def list_section(title, items, key=None, formatter=None):
        if not items:
            return
        print(f"--- {title} ---")
        for f in items:
            detail = ""
            if key:
                val = f[key]
                detail = f"  {formatter(val) if formatter else val}"
            print(f"  {f['path']}{detail}")
        print()

    list_section("Guarded imports", with_guarded, "guarded_imports")
    list_section("Dynamic import warnings", with_dyn_warn, "dynamic_warnings")
    list_section("%%writefile cells", with_writefile, "writefile_cells")
    list_section("Shell cells (%%bash etc)", with_shell, "shell_cells")
    list_section("Unhandled cell magics", with_unhandled_magic, "unhandled_magics")
    list_section("Magic warnings", with_magic_warn, "magic_warnings")
    list_section("Magic notices", with_magic_notice, "magic_notices")
    list_section("Hardware-tagged versions", with_hw_tags, "hardware_tags")
    list_section("Relative imports", with_rel_import)
    list_section("Parse errors", with_parse_error, "parse_error")
    list_section("Python 2.x notebooks", py2_notebooks, "python_version")


if __name__ == "__main__":
    main()
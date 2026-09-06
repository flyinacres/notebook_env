"""
Independent Corpus Inventory & Feature Audit (Phase 5j).
Scans a directory of .ipynb files directly (without depending on notebook_env's
own extraction pipeline) to measure real-world feature distribution and surface
unhandled magics, complex packaging flags, sibling files, and mapping gaps.

Usage:
    python diagnose_coverage.py [root_dir]     # default: test_notebooks
"""

import ast
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    import notebook_env as ne
    HAS_NE = True
except ImportError:
    HAS_NE = False

SKIP_DIRS = {".git", ".ipynb_checkpoints", "venv", "env", "__pycache__", ".tox", ".pytest_cache"}
KNOWN_SIBLING_CONFIGS = {
    "requirements.txt", "pyproject.toml", "setup.py", "setup.cfg",
    "environment.yml", "environment.yaml", "Pipfile", "Pipfile.lock"
}
STDLIB_MODULES = set(sys.stdlib_module_names) if hasattr(sys, "stdlib_module_names") else set()

# Regex patterns for independent scanning
HARDWARE_TAG_PATTERN = re.compile(r'[\w\-]+==[\d.]+\+[\w]+')
RELATIVE_IMPORT_PATTERN = re.compile(r'^\s*from\s+\.+\s*(?:\.\w+)*\s+import\s', re.MULTILINE)
PIP_LINE_PATTERN = re.compile(r'^\s*[%!]?\s*pip(?:3)?\s+install\s+(.*)$', re.MULTILINE)
SHELL_EXEC_PATTERN = re.compile(r'^\s*!(.*)$', re.MULTILINE)
MAGIC_LINE_PATTERN = re.compile(r'^\s*%(%?\w+)(?:\s+(.*))?$', re.MULTILINE)


def find_notebooks(root: Path):
    for path in sorted(root.rglob("*.ipynb")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def extract_ast_imports(code_text: str):
    """Parses pure python strings for top-level import roots and guarded imports."""
    imports = set()
    guarded = set()
    dynamic = []

    try:
        tree = ast.parse(code_text)
    except SyntaxError:
        return imports, guarded, dynamic

    for node in ast.walk(tree):
        # Detect try/except guarded imports
        if isinstance(node, ast.Try):
            for handler in node.handlers:
                # Check for broad or ImportError/ModuleNotFoundError handling
                catches_import = False
                if handler.type is None:
                    catches_import = True
                elif isinstance(handler.type, ast.Name) and handler.type.id in {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}:
                    catches_import = True
                elif isinstance(handler.type, ast.Tuple):
                    for elt in handler.type.elts:
                        if isinstance(elt, ast.Name) and elt.id in {"ImportError", "ModuleNotFoundError"}:
                            catches_import = True

                if catches_import:
                    for body_item in node.body:
                        if isinstance(body_item, ast.Import):
                            for alias in body_item.names:
                                guarded.add(alias.name.split('.')[0])
                        elif isinstance(body_item, ast.ImportFrom) and body_item.module:
                            guarded.add(body_item.module.split('.')[0])

        # Standard top-level import gathering
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split('.')[0])

        # Dynamic import warning inspection
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "import_module":
                dynamic.append(ast.unparse(node) if hasattr(ast, "unparse") else "import_module(...)")
            elif isinstance(node.func, ast.Name) and node.func.id == "__import__":
                dynamic.append(ast.unparse(node) if hasattr(ast, "unparse") else "__import__(...)")

    return imports, guarded, dynamic


def audit_notebook(path: Path) -> dict:
    findings = {
        "path": str(path),
        "parse_error": None,
        "authoring_platform": "standard",
        "python_version": None,
        "sibling_configs": set(),
        "has_local_py_helpers": False,
        "imports": set(),
        "guarded_imports": set(),
        "dynamic_imports": [],
        "cell_magics": set(),
        "line_magics": set(),
        "pip_commands": [],
        "git_installs": [],
        "editable_installs": [],
        "index_urls": [],
        "hardware_tags": set(),
        "relative_imports": False,
        "other_package_managers": set(),
    }

    # Inspect parent directory for sibling ecosystem files
    parent = path.parent
    for item in parent.iterdir():
        if item.is_file():
            if item.name in KNOWN_SIBLING_CONFIGS:
                findings["sibling_configs"].add(item.name)
            if item.suffix == ".py" and item.name != path.name:
                findings["has_local_py_helpers"] = True

    try:
        nb_data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        findings["parse_error"] = f"JSON read failure: {e}"
        return findings

    # Authoring metadata fingerprinting
    metadata = nb_data.get("metadata", {})
    if "colab" in metadata:
        findings["authoring_platform"] = "colab"
    elif "kaggle" in metadata:
        findings["authoring_platform"] = "kaggle"
    elif "vscode" in metadata or "interpreter" in metadata:
        findings["authoring_platform"] = "vscode"
    elif "databricks" in metadata:
        findings["authoring_platform"] = "databricks"
    elif any("application/vnd.databricks.v1+cell" in c.get("metadata", {}) for c in nb_data.get("cells", [])):
        findings["authoring_platform"] = "databricks"

    py_ver = metadata.get("language_info", {}).get("version")
    findings["python_version"] = py_ver

    # Cell-by-cell inspection
    cells = nb_data.get("cells", [])
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue

        raw_source = "".join(cell.get("source", []))
        if not raw_source.strip():
            continue

        # Cell and Line Magics
        first_line = raw_source.strip().splitlines()[0]
        if first_line.startswith("%%"):
            magic_name = first_line.split()[0]
            findings["cell_magics"].add(magic_name)

        for line in raw_source.splitlines():
            line_str = line.strip()
            if line_str.startswith("%") and not line_str.startswith("%%"):
                findings["line_magics"].add(line_str.split()[0])

            # Detect other package managers
            if re.match(r'^[%!]?\s*(conda|mamba|uv|poetry|pipenv)\s+', line_str):
                mgr = re.findall(r'(conda|mamba|uv|poetry|pipenv)', line_str)[0]
                findings["other_package_managers"].add(mgr)

        # Pip analysis
        for match in PIP_LINE_PATTERN.finditer(raw_source):
            cmd_args = match.group(1)
            findings["pip_commands"].append(cmd_args)
            if "git+" in cmd_args:
                findings["git_installs"].append(cmd_args)
            if "-e " in cmd_args or "--editable" in cmd_args:
                findings["editable_installs"].append(cmd_args)
            if "--index-url" in cmd_args or "--extra-index-url" in cmd_args or "-i " in cmd_args:
                findings["index_urls"].append(cmd_args)

        # Hardware tags
        for hw in HARDWARE_TAG_PATTERN.findall(raw_source):
            findings["hardware_tags"].add(hw)

        # Relative imports
        if RELATIVE_IMPORT_PATTERN.search(raw_source):
            findings["relative_imports"] = True

        # AST Inspection
        imp, guard, dyn = extract_ast_imports(raw_source)
        findings["imports"].update(imp)
        findings["guarded_imports"].update(guard)
        findings["dynamic_imports"].extend(dyn)

    return findings


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("test_notebooks")
    if not root.exists():
        print(f"Directory not found: {root}")
        sys.exit(1)

    notebooks = list(find_notebooks(root))
    print(f"Auditing {len(notebooks)} notebooks under {root}...\n")

    audits = [audit_notebook(p) for p in notebooks]

    # Aggregate Metrics
    parse_errors = [a for a in audits if a["parse_error"]]
    platforms = Counter(a["authoring_platform"] for a in audits)
    other_pkg_mgrs = Counter(mgr for a in audits for mgr in a["other_package_managers"])
    all_cell_magics = Counter(m for a in audits for m in a["cell_magics"])
    all_line_magics = Counter(m for a in audits for m in a["line_magics"])
    with_local_helpers = [a for a in audits if a["has_local_py_helpers"]]
    with_sibling_configs = [a for a in audits if a["sibling_configs"]]
    with_git = [a for a in audits if a["git_installs"]]
    with_editable = [a for a in audits if a["editable_installs"]]
    with_index = [a for a in audits if a["index_urls"]]
    with_hw_tags = [a for a in audits if a["hardware_tags"]]
    with_rel_import = [a for a in audits if a["relative_imports"]]
    with_guarded = [a for a in audits if a["guarded_imports"]]
    with_dyn = [a for a in audits if a["dynamic_imports"]]

    # Unmapped / Unresolved imports check
    import_counts = Counter(imp for a in audits for imp in a["imports"])
    known_mappings = getattr(ne, "IMPORT_TO_PYPI_MAP", {}) if HAS_NE else {}
    pseudo_modules = {"google", "kaggle_secrets", "kaggle_web_client", "IPython", "ipywidgets"}

    unresolved_imports = {}
    for imp, count in import_counts.items():
        if imp in STDLIB_MODULES or imp in pseudo_modules:
            continue
        if imp not in known_mappings:
            unresolved_imports[imp] = count

    print("=" * 70)
    print("PHASE 5J: CORPUS FEATURE AUDIT SUMMARY")
    print("=" * 70)
    print(f"Total Notebooks Analyzed:               {len(audits)}")
    print(f"JSON / Parse Errors:                    {len(parse_errors)}")
    print("\n--- Authoring Platforms Detected ---")
    for plat, count in platforms.items():
        print(f"  {plat:<20}: {count}")

    print("\n--- Packaging & Invocations ---")
    print(f"Git URL Installs (git+https):           {len(with_git)}")
    print(f"Editable Installs (-e / --editable):    {len(with_editable)}")
    print(f"Custom Index URLs (--extra-index-url):  {len(with_index)}")
    print(f"Hardware-Tagged Packages (+cu/etc):     {len(with_hw_tags)}")
    print(f"Alternative Package Managers:           {dict(other_pkg_mgrs)}")

    print("\n--- Structure & Ecosystem Context ---")
    print(f"Notebooks with Sibling .py Helpers:     {len(with_local_helpers)}")
    print(f"Notebooks with Sibling Config Files:    {len(with_sibling_configs)}")
    print(f"Relative Imports (from . import):       {len(with_rel_import)}")

    print("\n--- Code Mechanics & Magics ---")
    print(f"Guarded Imports (try/except):           {len(with_guarded)}")
    print(f"Dynamic Imports:                        {len(with_dyn)}")
    print(f"Distinct Cell Magics (%%):              {dict(all_cell_magics)}")
    print(f"Distinct Line Magics (%):               {dict(all_line_magics)}")

    if unresolved_imports:
        print("\n--- Top Unresolved Imports (Not in stdlib or IMPORT_TO_PYPI_MAP) ---")
        sorted_unresolved = sorted(unresolved_imports.items(), key=lambda x: x[1], reverse=True)[:25]
        for imp, count in sorted_unresolved:
            print(f"  {imp:<30}: found in {count} notebook(s)")
    print("=" * 70)


if __name__ == "__main__":
    main()
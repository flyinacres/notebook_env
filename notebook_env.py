#!/usr/bin/env python3
"""
notebook_env.py (v33)
Headless Jupyter Notebook Dependency Scanner & Lockfile Generator.

Standalone, zero-dependency utility for analyzing notebook environments,
detecting GPU/accelerator requirements, harvesting index URLs, and emitting
reproducible lockfile manifests (`pinned_requirements.txt`).

Execution Modes:
  1. Single Notebook CLI:  python notebook_env.py notebook.ipynb
  2. Batch Repo Directory: python notebook_env.py --batch ./repo --universal
  3. Live IPython Kernel:   import notebook_env as ne; ne.main()

For full usage, CLI flag documentation, and architectural details, see README.md.
"""

# =====================================================================
# CONSTANTS, LOGGING & TYPE DEFINITIONS
# Core data structures (ScanResult, GpuInfo), logging setup directed to 
# stderr, status labels, and static mapping dicts (IMPORT_TO_PYPI_MAP, STD_LIB).
# =====================================================================

import ast
import json
import os
import re
import sys
import argparse
import logging
import subprocess
import warnings
import importlib.metadata
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Set, Dict, List, Tuple, Optional, Any, TypedDict


# Force UTF-8 encoding for stdout and stderr on Windows/redirected environments
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Setup logging stream for diagnostic messages (directed to stderr)
logger = logging.getLogger("notebook_env")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stderr)
console_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(console_handler)


class StatusLabel:
    PYTHON = "python"
    CORRUPTED = "corrupted"
    ERROR = "error"
    UNKNOWN = "unknown"
    MISSING_METADATA = "missing metadata"


class GpuInfo(TypedDict, total=False):
    has_gpu: bool
    type: Optional[str]
    active_framework: Optional[str]
    device_name: Optional[str]
    frameworks: List[str]


class BlueprintResult(TypedDict):
    step1_markdown: str
    step2_code: str


@dataclass
class NotebookScanResult:
    path: Path
    is_python: bool
    lang_label: str
    parse_error: Optional[str] = None
    imports: Set[str] = field(default_factory=set)
    submodules: Dict[str, Set[str]] = field(default_factory=dict)
    guarded_imports: Set[str] = field(default_factory=set)
    dynamic_warnings: List[str] = field(default_factory=list)
    code_sources: List[str] = field(default_factory=list)
    harvested_urls: Set[str] = field(default_factory=set)
    writefile_imports: Set[str] = field(default_factory=set)
    harvested_pkgs: Set[str] = field(default_factory=set)
    base_index_urls: Set[str] = field(default_factory=set)
    extra_index_urls: Set[str] = field(default_factory=set)
    magic_warnings: List[str] = field(default_factory=list)
    magic_notices: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.code_sources and not self.harvested_urls:
            self.harvested_urls = harvest_index_urls_from_sources(self.code_sources)


# Mappings, Platform Injections & Stdlib lookup
IMPORT_TO_PYPI_MAP: Dict[str, str] = {
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "PIL": "Pillow",
    "yaml": "PyYAML",
    "bs4": "beautifulsoup4",
    "attr": "attrs",
    "serial": "pyserial",
    "dotenv": "python-dotenv",
    "mpl_toolkits": "matplotlib"
}

PLATFORM_PSEUDO_MODULES: Set[str] = {
    "dbutils",
    "kaggle_secrets",
    "google.colab",
    "pyspark.dbutils"
}

STD_LIB: Set[str] = set(sys.stdlib_module_names) if hasattr(sys, 'stdlib_module_names') else {
    "os", "sys", "re", "json", "ast", "subprocess", "datetime", "math", "random", 
    "time", "pathlib", "typing", "collections", "itertools", "functools", "shutil"
}


def discover_local_repo_modules(target_dir: str) -> Set[str]:
    """Scans target_dir for local .py files and package directories containing __init__.py."""
    local_mods: Set[str] = set()
    target_path = Path(target_dir)
    if not target_path.exists():
        return local_mods

    try:
        for entry in target_path.iterdir():
            if entry.is_file() and entry.suffix == ".py":
                local_mods.add(entry.stem)
            elif entry.is_dir() and (entry / "__init__.py").exists():
                local_mods.add(entry.name)
    except Exception:
        pass

    return local_mods


# =====================================================================
# CELL CLASSIFICATION & SOURCE PIPELINE
# Ingestion entrypoints (extract_from_file for saved notebooks, 
# extract_from_active_session for live kernels) and kernel language metadata checks.
# =====================================================================

def detect_notebook_language(nb_data: Dict[str, Any], strict: bool = False) -> Tuple[bool, str]:
    """
    Inspects kernelspec and language_info metadata.
    If strict=True (batch mode), missing metadata is rejected as unknown.
    If strict=False (single-file mode), missing metadata assumes Python.
    """
    metadata = nb_data.get("metadata", {})
    ks_lang = metadata.get("kernelspec", {}).get("language", "").lower()
    li_lang = metadata.get("language_info", {}).get("name", "").lower()

    if ks_lang and li_lang:
        if ks_lang == li_lang:
            return (ks_lang == StatusLabel.PYTHON), ks_lang
        else:
            return False, f"conflict ({ks_lang}/{li_lang})"
    
    active_lang = ks_lang or li_lang
    if active_lang:
        return (active_lang == StatusLabel.PYTHON), active_lang
        
    return True, "unspecified (assuming python)"


def extract_from_file(
    notebook_path: str, strict: bool = False
) -> Tuple[bool, Set[str], Dict[str, Set[str]], List[str], Optional[str], str, Set[str], List[str]]:
    """Reads a Jupyter Notebook JSON file and extracts code sources, imports, guarded state, and dynamic warnings."""
    if not os.path.exists(notebook_path):
        return False, set(), {}, [], f"File '{notebook_path}' not found.", StatusLabel.UNKNOWN, set(), []

    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb_data = json.load(f)
    except json.JSONDecodeError as e:
        return False, set(), {}, [], f"Invalid JSON structure ({e})", StatusLabel.CORRUPTED, set(), []
    except Exception as e:
        return False, set(), {}, [], f"File read failure ({e})", StatusLabel.ERROR, set(), []

    if not isinstance(nb_data, dict) or "cells" not in nb_data or not isinstance(nb_data.get("cells"), list):
        return False, set(), {}, [], "Unparseable notebook structure (Missing or invalid 'cells' array)", StatusLabel.CORRUPTED, set(), []

    is_py, lang_label = detect_notebook_language(nb_data, strict=strict)
    if not is_py:
        return False, set(), {}, [], f"Skipped non-Python notebook (Language: {lang_label})", lang_label, set(), []

    cells = nb_data.get("cells", [])
    code_sources = ["".join(c.get("source", [])) for c in cells if c.get("cell_type") == "code"]
    imports, submodules, guarded_imports, dyn_warnings = extract_imports_from_sources(code_sources)
    return True, imports, submodules, code_sources, None, lang_label, guarded_imports, dyn_warnings


def extract_from_active_session() -> Tuple[Set[str], Dict[str, Set[str]], List[str], Set[str], List[str]]:
    """Path B (Live Kernel): Reads IPython execution history."""
    import __main__
    code_sources = [src for src in getattr(__main__, 'In', []) if src and isinstance(src, str)]
    imports, submodules, code_sources, guarded_imports, dyn_warnings = extract_from_active_session_internal(code_sources)
    return imports, submodules, code_sources, guarded_imports, dyn_warnings


def extract_from_active_session_internal(code_sources: List[str]) -> Tuple[Set[str], Dict[str, Set[str]], List[str], Set[str], List[str]]:
    imports, submodules, guarded_imports, dyn_warnings = extract_imports_from_sources(code_sources)
    return imports, submodules, code_sources, guarded_imports, dyn_warnings


# =====================================================================
# AST VISITOR & DYNAMIC IMPORT PARSER
# Python AST traversal engine. Inspects import statements, submodules,
# guarded try/except blocks, and importlib/literal dynamic import calls.
# =====================================================================

class NotebookImportVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.imports: Set[str] = set()
        self.writefile_imports: Set[str] = set()
        self.submodules: Dict[str, Set[str]] = {}
        self.unconditional_imports: Set[str] = set()
        self.raw_guarded_imports: Set[str] = set()
        self.dynamic_import_warnings: List[str] = []
        self._guarded_depth: int = 0
        self._in_writefile: bool = False

        # Track bound aliases for importlib and import_module
        self._importlib_aliases: Set[str] = {"importlib"}
        self._import_module_bindings: Set[str] = set()

    @property
    def guarded_imports(self) -> Set[str]:
        return self.raw_guarded_imports - self.unconditional_imports

    def _record_import(self, base_pkg: str, full_name: Optional[str] = None) -> None:
        if self._in_writefile:
            self.writefile_imports.add(base_pkg)
            return

        self.imports.add(base_pkg)
        if self._guarded_depth > 0:
            self.raw_guarded_imports.add(base_pkg)
        else:
            self.unconditional_imports.add(base_pkg)

        if full_name and '.' in full_name:
            self.submodules.setdefault(base_pkg, set()).add(full_name)

    def visit_Try(self, node: ast.Try) -> None:
        self._guarded_depth += 1
        self.generic_visit(node)
        self._guarded_depth -= 1

    def visit_If(self, node: ast.If) -> None:
        self._guarded_depth += 1
        self.generic_visit(node)
        self._guarded_depth -= 1

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            base_pkg = alias.name.split('.')[0]
            if alias.name == "importlib":
                self._importlib_aliases.add(alias.asname or "importlib")
            self._record_import(base_pkg, full_name=alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            base_pkg = node.module.split('.')[0]
            if node.module == "importlib":
                for alias in node.names:
                    if alias.name == "import_module":
                        self._import_module_bindings.add(alias.asname or "import_module")
            self._record_import(base_pkg, full_name=node.module)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        is_dynamic_import = False

        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id in self._importlib_aliases:
                if node.func.attr == "import_module":
                    is_dynamic_import = True

        elif isinstance(node.func, ast.Name):
            if node.func.id == "__import__" or node.func.id in self._import_module_bindings:
                is_dynamic_import = True

        if is_dynamic_import and node.args:
            first_arg = node.args[0]

            if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                imported_pkg = first_arg.value
                base_pkg = imported_pkg.split('.')[0]
                self._record_import(base_pkg, full_name=imported_pkg)
            else:
                expr_repr = ast.unparse(first_arg) if hasattr(ast, "unparse") else "expression"
                self.dynamic_import_warnings.append(
                    f"⚠️ Dynamic import detected via non-literal argument '{expr_repr}'; statically unresolvable."
                )

        self.generic_visit(node)


def extract_imports_from_sources(
    code_sources: List[str]
) -> Tuple[Set[str], Dict[str, Set[str]], Set[str], List[str]]:
    visitor = NotebookImportVisitor()
    for source in code_sources:
        cell_type, clean_body = classify_cell_source(source)

        if cell_type == "SHELL_SCRIPT":
            continue

        visitor._in_writefile = (cell_type == "WRITEFILE")

        clean_source = "\n".join([
            line for line in clean_body.splitlines() 
            if not line.strip().startswith('%') and not line.strip().startswith('!')
        ])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=SyntaxWarning)
                tree = ast.parse(clean_source)
            visitor.visit(tree)
        except SyntaxError:
            continue

    primary_imports = visitor.imports - visitor.writefile_imports

    return (
        primary_imports, 
        visitor.submodules, 
        visitor.guarded_imports, 
        visitor.dynamic_import_warnings
    )

def extract_writefile_imports_from_sources(code_sources: List[str]) -> Set[str]:
    visitor = NotebookImportVisitor()
    for source in code_sources:
        cell_type, clean_body = classify_cell_source(source)
        if cell_type == "WRITEFILE":
            visitor._in_writefile = True
            clean_source = "\n".join([
                line for line in clean_body.splitlines() 
                if not line.strip().startswith('%') and not line.strip().startswith('!')
            ])
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=SyntaxWarning)
                    tree = ast.parse(clean_source)
                visitor.visit(tree)
            except SyntaxError:
                continue
    return visitor.writefile_imports


# =====================================================================
# CELL MAGIC & SHELL COMMAND HARVESTER
# Regex and line scanners for IPython cell magics (%pip, !pip, %conda),
# base/extra PyPI index URLs (--index-url), and non-Python shell commands.
# =====================================================================

PIP_SINGLE_FLAGS: Set[str] = {
    "-u", "--upgrade", "-q", "--quiet", "--user", "--no-cache-dir",
    "--force-reinstall", "--no-deps", "--pre", "--break-system-packages"
}

PIP_VALUE_FLAGS: Set[str] = {
    "--extra-index-url", "--index-url", "-i", "-f", "--find-links", 
    "-t", "--target", "-e", "--editable", "-r", "--requirement"
}

SHELL_CELL_MAGICS: Set[str] = {
    "%%bash", "%%sh", "%%zsh", "%%script", "%%cmd", "%%powershell"
}

# --- COMPILED REGEX PATTERNS (Section-Level Constants) ---
EXTRA_INDEX_PATTERN = re.compile(r'--extra-index-url\s+([^\s]+)')
BASE_INDEX_PATTERN = re.compile(r'(?:--index-url|-i)\s+([^\s]+)')
SHELL_SPLIT_PATTERN = re.compile(r'\s*(?:&&|;|\||\|\|)\s*')

PIP_INSTALL_PATTERN = re.compile(r'^\s*(?:%pip|!pip|pip3?)\s+install\s+(.+)$')
SYSTEM_PKG_PATTERN = re.compile(r'^\s*(?:!|%%bash|%%sh)?\s*(?:apt-get|brew|yum)\s+install\s+(.+)$')
CONDA_INSTALL_PATTERN = re.compile(r'^\s*(?:%conda|!conda|conda)\s+install\s+(.+)$')

VCS_OR_PATH_PREFIXES: Tuple[str, ...] = (
    ".", "/", "\\", "git+", "hg+", "svn+", "bzr+", "http://", "https://"
)


def classify_cell_source(source: str) -> Tuple[str, str]:
    """Classifies cell source into (cell_type, clean_source)."""
    lines = source.splitlines()
    if not lines:
        return "PYTHON", ""

    first_line = lines[0].strip()
    first_token = first_line.split()[0] if first_line.split() else ""

    if first_token in SHELL_CELL_MAGICS:
        return "SHELL_SCRIPT", "\n".join(lines[1:])

    if first_token == "%%writefile":
        return "WRITEFILE", "\n".join(lines[1:])

    return "PYTHON", source


def harvest_index_urls_from_sources(code_sources: List[str]) -> Set[str]:
    """
    Scans code sources for index URLs and returns a combined set of all harvested URLs.
    Preserves signature compatibility for batch runners and existing test suites.
    """
    _, base_urls, extra_urls, _, _ = harvest_cell_magics_and_commands(code_sources)
    return base_urls.union(extra_urls)


def harvest_cell_magics_and_commands(
    code_sources: List[str]
) -> Tuple[Set[str], Set[str], Set[str], List[str], List[str]]:
    """
    Scans code sources for cell magics, index URLs, auxiliary tools, and shell commands.

    Returns:
        - harvested_packages: Set[str] (auxiliary tools installed via %pip / !pip)
        - base_index_urls: Set[str] (--index-url / -i base index overrides)
        - extra_index_urls: Set[str] (--extra-index-url supplemental indexes)
        - magic_warnings: List[str] (warnings for -r requirements.txt or unresolvable scripts)
        - magic_notices: List[str] (informational notices for conda / apt-get calls)
    """
    harvested_packages: Set[str] = set()
    base_index_urls: Set[str] = set()
    extra_index_urls: Set[str] = set()
    magic_warnings: List[str] = []
    magic_notices: List[str] = []

    for cell_idx, source in enumerate(code_sources, start=1):
        for line in source.splitlines():
            clean_line = line.strip()
            if not clean_line or clean_line.startswith('#') or clean_line in SHELL_CELL_MAGICS:
                continue

            # 1. Harvest Index URLs (Token-aware to prevent cross-flag pollution)
            for match in EXTRA_INDEX_PATTERN.finditer(clean_line):
                extra_index_urls.add(match.group(1).strip("'\""))
            
            for match in BASE_INDEX_PATTERN.finditer(clean_line):
                full_match_str = match.group(0)
                if not full_match_str.startswith("--extra-index-url"):
                    base_index_urls.add(match.group(1).strip("'\""))

            # 2. Split chained commands (&&, ;) and evaluate segments
            command_segments = SHELL_SPLIT_PATTERN.split(clean_line)

            for segment in command_segments:
                seg = segment.strip()
                if not seg:
                    continue

                if SYSTEM_PKG_PATTERN.match(seg):
                    magic_notices.append(
                        f"ℹ️ System package manager call detected in cell {cell_idx} ('{seg}'); "
                        f"system-level dependencies are outside Python package manifests."
                    )
                    continue

                if CONDA_INSTALL_PATTERN.match(seg):
                    magic_notices.append(
                        f"ℹ️ Conda installation detected in cell {cell_idx} ('{seg}'); "
                        f"conda packages are outside pip freeze correlation."
                    )
                    continue

                pip_match = PIP_INSTALL_PATTERN.match(seg)
                if pip_match:
                    args_str = pip_match.group(1)

                    if "-r " in args_str or "--requirement" in args_str:
                        magic_warnings.append(
                            f"⚠️ Cell {cell_idx} magic '{seg}' references an external requirements file; "
                            f"contents cannot be verified statically."
                        )
                        continue

                    tokens = args_str.split()
                    i = 0
                    while i < len(tokens):
                        token = tokens[i]
                        
                        if token in PIP_VALUE_FLAGS:
                            i += 2
                            continue
                        
                        if token.startswith('-') or token.lower() in PIP_SINGLE_FLAGS:
                            i += 1
                            continue

                        if any(token.lower().startswith(prefix) for prefix in VCS_OR_PATH_PREFIXES):
                            i += 1
                            continue

                        pkg_name = re.split(r'[<>=!~;\[#]', token)[0].strip("'\"")
                        if pkg_name:
                            harvested_packages.add(pkg_name)
                        i += 1

    return harvested_packages, base_index_urls, extra_index_urls, magic_warnings, magic_notices


# =====================================================================
# ENVIRONMENT CORRELATION & EXTRAS PROMOTION
# Maps extracted import names to active runtime versions via `pip freeze` 
# and importlib metadata. Handles optional extras promotion (e.g., umap.plot -> umap-learn[plot]).
# =====================================================================

def build_auxiliary_tool_entries(
    harvested_packages: Set[str],
    imported_packages: Set[str],
    frozen_env: Dict[str, str]
) -> List[str]:
    """
    Builds commented-out manifest lines for CLI tools installed via cell magics
    that are not directly imported in Python code.
    """
    aux_entries: List[str] = []
    unimported_tools = sorted([
        pkg for pkg in harvested_packages 
        if pkg.lower() not in {imp.lower() for imp in imported_packages} and pkg.lower() not in STD_LIB
    ])

    if not unimported_tools:
        return aux_entries

    aux_entries.append("\n# --- AUXILIARY TOOL INSTALLS (harvested from cell magics) ---")
    for tool in unimported_tools:
        matched_pin = frozen_env.get(tool.lower())
        if matched_pin:
            aux_entries.append(f"# {matched_pin}  (installed via cell magic; not directly imported in Python code)")
        else:
            aux_entries.append(f"# {tool}  (installed via cell magic; not found in active env)")

    return aux_entries


def build_writefile_tool_entries(
    writefile_imports: Set[str],
    primary_imports: Set[str],
    frozen_env: Dict[str, str]
) -> List[str]:
    """
    Builds commented-out manifest lines for dependencies imported exclusively
    inside %%writefile generated scripts.
    """
    entries: List[str] = []
    script_only = sorted([
        pkg for pkg in writefile_imports 
        if pkg.lower() not in {imp.lower() for imp in primary_imports} and pkg.lower() not in STD_LIB
    ])

    if not script_only:
        return entries

    entries.append("\n# --- WRITEFILE SCRIPT DEPENDENCIES ---")
    for pkg in script_only:
        pypi_name = IMPORT_TO_PYPI_MAP.get(pkg, pkg)
        matched_pin = frozen_env.get(pypi_name.lower())
        if matched_pin:
            entries.append(f"# {matched_pin}  (imported inside script generated via %%writefile)")
        else:
            entries.append(f"# {pypi_name}  (imported inside script generated via %%writefile; not found in active env)")

    return entries


def resolve_pypi_package_and_extras(
    imp: str, 
    submodules_set: Set[str], 
    frozen_env: Dict[str, str], 
    pkg_dist_map: Optional[Dict[str, List[str]]] = None,
    is_guarded: bool = False,
    local_repo_modules: Optional[Set[str]] = None
) -> Tuple[str, Optional[str]]:
    """Resolves top-level import to PyPI package name, platform pseudo-module, or local repo module."""
    if imp in PLATFORM_PSEUDO_MODULES:
        return f"# {imp} (platform pseudo-module provided by runtime environment)", None

    if local_repo_modules and imp in local_repo_modules:
        return f"# {imp} (local repo module; not a PyPI package)", None

    pypi_name = None
    if pkg_dist_map is None and hasattr(importlib.metadata, "packages_distributions"):
        try:
            pkg_dist_map = importlib.metadata.packages_distributions()
        except Exception:
            pkg_dist_map = {}

    if pkg_dist_map and imp in pkg_dist_map:
        pypi_name = pkg_dist_map[imp][0]

    if imp == "cv2":
        pypi_name = resolve_opencv_variant(submodules_set)

    if not pypi_name:
        pypi_name = IMPORT_TO_PYPI_MAP.get(imp, imp)

    matched_pin = frozen_env.get(pypi_name.lower())

    if is_guarded:
        if matched_pin:
            return f"# {matched_pin} (guarded import in try/except or conditional block - optional dependency)", None
        return f"# {pypi_name} (imported as '{imp}' in try/except or conditional block - optional fallback)", None

    if not matched_pin:
        return f"# {pypi_name} (imported as '{imp}', not currently found in active env)", None

    pkg_part, ver_part = matched_pin.split("==", 1)

    extra_tag = None
    if submodules_set:
        try:
            dist = importlib.metadata.distribution(pkg_part)
            provided_extras = dist.metadata.get_all("Provides-Extra") or []
            provided_extras_lower = {e.lower(): e for e in provided_extras}

            for sub in submodules_set:
                sub_tail = sub.split('.')[-1].lower()
                if sub_tail in provided_extras_lower:
                    extra_tag = provided_extras_lower[sub_tail]
                    break
        except Exception:
            pass

    if extra_tag:
        promoted_pin = f"{pkg_part}[{extra_tag}]=={ver_part}"
        notice = f"💡 Extra Dependency Promotion: importing '{imp}.{extra_tag}' automatically promoted requirement to '{pkg_part}[{extra_tag}]=={ver_part}'"
        return promoted_pin, notice

    return matched_pin, None


def build_manifest_entries(
    imports: Set[str], 
    submodules: Dict[str, Set[str]], 
    frozen_env: Dict[str, str], 
    pkg_dist_map: Optional[Dict[str, List[str]]] = None,
    guarded_imports: Optional[Set[str]] = None,
    local_repo_modules: Optional[Set[str]] = None
) -> Tuple[List[str], List[str]]:
    """Single shared helper for generating correlated pinned manifest entries."""
    pinned_manifest: List[str] = []
    promotion_notices: List[str] = []
    guarded_set = guarded_imports or set()

    for imp in sorted(imports):
        if imp in STD_LIB:
            continue
        submods = submodules.get(imp, set())
        is_guarded = imp in guarded_set
        pin_entry, notice = resolve_pypi_package_and_extras(
            imp, submods, frozen_env, pkg_dist_map=pkg_dist_map, is_guarded=is_guarded, local_repo_modules=local_repo_modules
        )
        pinned_manifest.append(pin_entry)
        if notice and notice not in promotion_notices:
            promotion_notices.append(notice)

    return pinned_manifest, promotion_notices


def resolve_opencv_variant(submodules: Optional[Set[str]] = None) -> str:
    """Determines the appropriate OpenCV package variant installed in the environment."""
    has_contrib = any('contrib' in s.lower() or 'aruco' in s.lower() for s in submodules) if submodules else False
    try:
        res = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True)
        installed = res.stdout.lower()
        if "opencv-contrib-python-headless" in installed:
            return "opencv-contrib-python-headless"
        elif "opencv-python-headless" in installed:
            return "opencv-python-headless"
        elif "opencv-contrib-python" in installed:
            return "opencv-contrib-python"
        elif "opencv-python" in installed:
            return "opencv-python"
    except Exception:
        pass
    return "opencv-contrib-python" if has_contrib else "opencv-python"


def get_installed_environment() -> Tuple[Dict[str, str], List[str]]:
    """Runs pip freeze to get precise version snapshots."""
    res = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    frozen: Dict[str, str] = {}
    for line in res.stdout.splitlines():
        if "==" in line:
            pkg, ver = line.split("==", 1)
            frozen[pkg.lower()] = line.strip()
    return frozen, res.stdout.splitlines()


def process_package_requirements(
    pinned_list: List[str], 
    harvested_urls: Set[str],
    base_urls: Optional[Set[str]] = None,
    auxiliary_entries: Optional[List[str]] = None,
    writefile_entries: Optional[List[str]] = None
) -> Tuple[List[str], List[Tuple[str, List[str]]], List[str]]:
    """Correlates pinned packages with harvested base/extra index URLs, auxiliary tool entries, and writefile script dependencies."""
    manifest_output: List[str] = []
    local_tagged_info: List[Tuple[str, List[str]]] = []
    warnings: List[str] = []
    
    # 1. Base index URL overrides (--index-url)
    if base_urls:
        for url in sorted(base_urls):
            manifest_output.append(f"--index-url {url}")

    # 2. Extra index URLs (--extra-index-url)
    extra_urls = harvested_urls - (base_urls or set())
    if extra_urls:
        for url in sorted(extra_urls):
            manifest_output.append(f"--extra-index-url {url}")

    # 3. Primary imported dependencies
    for item in pinned_list:
        manifest_output.append(item)
        if '+' in item:
            all_urls = sorted(harvested_urls.union(base_urls or set()))
            local_tagged_info.append((item, all_urls))
            if not all_urls:
                warnings.append(item)

    # 4. Auxiliary tool section
    if auxiliary_entries:
        manifest_output.extend(auxiliary_entries)

    # 5. Writefile script dependency section
    if writefile_entries:
        manifest_output.extend(writefile_entries)
            
    return manifest_output, local_tagged_info, warnings


# =====================================================================
# HARDWARE ACCELERATION INSPECTION
# Probes runtime framework state for GPU/accelerator availability across 
# PyTorch (CUDA/MPS), TensorFlow (GPU), and JAX (GPU/TPU).
# =====================================================================

def inspect_gpu_environment(imported_packages: Set[str]) -> Optional[GpuInfo]:
    """Per-framework GPU/accelerator inspection logic."""
    gpu_frameworks = {"torch", "tensorflow", "jax"}
    found_frameworks = list(gpu_frameworks.intersection(imported_packages))
    
    if not found_frameworks:
        return None

    if "torch" in found_frameworks:
        try:
            import torch
            if torch.cuda.is_available():
                return {
                    "has_gpu": True,
                    "type": "NVIDIA CUDA",
                    "active_framework": "PyTorch",
                    "device_name": f"{torch.cuda.get_device_name(0)} (via PyTorch)",
                    "frameworks": found_frameworks
                }
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return {
                    "has_gpu": True,
                    "type": "Apple Silicon MPS",
                    "active_framework": "PyTorch",
                    "device_name": "Apple Silicon GPU (Metal via PyTorch)",
                    "frameworks": found_frameworks
                }
        except Exception:
            pass

    if "tensorflow" in found_frameworks:
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                dev_name = "NVIDIA GPU (via TensorFlow)"
                try:
                    details = tf.config.experimental.get_device_details(gpus[0])
                    dev_name = f"{details.get('device_name', 'NVIDIA GPU')} (via TensorFlow)"
                except Exception:
                    pass
                return {
                    "has_gpu": True,
                    "type": "GPU",
                    "active_framework": "TensorFlow",
                    "device_name": dev_name,
                    "frameworks": found_frameworks
                }
        except Exception:
            pass

    if "jax" in found_frameworks:
        try:
            import jax
            devices = jax.devices()
            accelerators = [d for d in devices if d.platform in ("gpu", "tpu")]
            if accelerators:
                first_accel = accelerators[0]
                accel_type = first_accel.platform.upper()
                dev_name = f"{accel_type} ({first_accel.device_kind}) via JAX"
                return {
                    "has_gpu": True,
                    "type": accel_type,
                    "active_framework": "JAX",
                    "device_name": dev_name,
                    "frameworks": found_frameworks
                }
        except Exception:
            pass

    return {
        "has_gpu": False,
        "type": None,
        "active_framework": None,
        "device_name": None,
        "frameworks": found_frameworks
    }


# =====================================================================
# BLUEPRINT & MANAGED SETUP CELL GENERATOR
# Constructs the self-contained Cell 1 (Markdown setup guide) and 
# Cell 2 (Python verification script that writes pinned_requirements.txt and runs pip install).
# =====================================================================

def generate_production_blueprint(
    manifest_lines: List[str], 
    full_freeze_lines: Optional[List[str]] = None, 
    local_tagged_info: Optional[List[Tuple[str, List[str]]]] = None, 
    gpu_info: Optional[GpuInfo] = None
) -> BlueprintResult:
    """Assembles Cell 1 Markdown and Cell 2 Python code dictionary."""
    py_major, py_minor = sys.version_info.major, sys.version_info.minor
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    gpu_markdown_section = ""
    if gpu_info and gpu_info.get("has_gpu"):
        dev_name = gpu_info["device_name"]
        active_fw = gpu_info.get("active_framework", "Framework")
        gpu_markdown_section = (
            f"- **Hardware Acceleration:** This notebook was created using an active accelerator (`{dev_name}`, verified via {active_fw}).\n"
            f"  If execution is slow or fails, you MAY need to enable a GPU accelerator in your environment settings (e.g. CUDA/MPS/TPU)."
        )

    local_builds_section = ""
    if local_tagged_info:
        bullet_lines = []
        for pkg, urls in local_tagged_info:
            bullet_lines.append(f"  • `{pkg}`")
            if urls:
                for u in urls:
                    bullet_lines.append(f"    Download index: `{u}`")
            else:
                bullet_lines.append("    ⚠️ No download URL was specified in notebook cells. If installation fails, ensure your target runtime matches this build or supply an `--extra-index-url`.")
        local_builds_section = f"- **Specific Package Builds Detected:** The following package(s) use custom or platform-specific local builds:\n" + "\n".join(bullet_lines)

    markdown_lines = [
        "### 🛠️ Environment Setup & Dependency Verification",
        "This notebook includes a pinned environment manifest (`pinned_requirements.txt`) to ensure reproducible execution.\n",
        "- **Dependency Sync:** Cell 2 will verify your active Python version and apply the exact package manifest recorded by the author."
    ]
    
    if gpu_markdown_section:
        markdown_lines.append(gpu_markdown_section)
    if local_builds_section:
        markdown_lines.append(local_builds_section)
        
    markdown_lines.append("- **Network Notice:** If required packages are not already cached in your current runtime environment, internet access may be needed to download missing wheels.")

    step1_markdown = "\n".join(markdown_lines)
    payload_string = "\n".join(manifest_lines).strip()
    if full_freeze_lines:
        payload_string += "\n\n# --- FULL FREEZE FALLBACK BLOCK ---\n" + "\n".join(full_freeze_lines)

    step2_code = f"""# =====================================================================
# VERIFIED ENVIRONMENT DEPENDENCIES ({timestamp})
# =====================================================================

import sys
import subprocess

REQUIRED_PYTHON = ({py_major}, {py_minor})
CURRENT_PYTHON = (sys.version_info.major, sys.version_info.minor)

# Major version mismatch -> Clean hard stop
if CURRENT_PYTHON[0] != REQUIRED_PYTHON[0]:
    req_major = REQUIRED_PYTHON[0]
    curr_major = CURRENT_PYTHON[0]
    print(f"❌ Error: Major Python version mismatch!")
    print(f"This notebook requires Python {{req_major}}.x, but your environment is running Python {{curr_major}}.x.\\n")
    sys.exit("Execution stopped due to Python major version incompatibility.")

# Minor version mismatch -> Non-blocking warning
if CURRENT_PYTHON[1] != REQUIRED_PYTHON[1]:
    req_ver = f"{{REQUIRED_PYTHON[0]}}.{{REQUIRED_PYTHON[1]}}"
    curr_ver = f"{{CURRENT_PYTHON[0]}}.{{CURRENT_PYTHON[1]}}"
    print(f"⚠️ This code was created with Python {{req_ver}}. You are trying to run it with {{curr_ver}}.")
    print(f"If installation fails, you may need to change your Python version back to {{req_ver}}.\\n")

# Write explicit library requirements to a local file
requirements_content = \"\"\"# Tested top-level packages for this notebook
{payload_string}
\"\"\"

with open("pinned_requirements.txt", "w") as f:
    f.write(requirements_content.strip())

print(f"Applying pinned environment manifest [{timestamp}]...")
print("💡 Note: If you see 'Retrying...' messages below while offline, enable Internet access and re-run this cell.\\n")

# Run single-pass installation natively via pip
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-r", "pinned_requirements.txt"],
    capture_output=False
)

if result.returncode == 0:
    print("\\n✅ Setup complete! Environment ready.")
else:
    print("\\n❌ Setup failed while installing pinned dependencies.")
    print("It looks like your environment could not locate a matching wheel for local tag builds (e.g. +cu121, +cpu).\\n")
    print("Possible solutions:")
    print("1. Make sure your notebook runtime matches the required hardware (e.g. GPU vs CPU).")
    print("2. Or try installing the standard version directly in a code cell (e.g. !pip install torch).")"""

    return {
        "step1_markdown": step1_markdown,
        "step2_code": step2_code
    }


def create_managed_cells(blueprint: BlueprintResult) -> List[Dict[str, Any]]:
    """Creates cell dicts stamped with notebook_env managed metadata."""
    cell1 = {
        "cell_type": "markdown",
        "metadata": {
            "notebook_env": {
                "managed": True,
                "role": "setup_markdown"
            }
        },
        "source": [line + "\n" for line in blueprint["step1_markdown"].splitlines()]
    }
    cell2 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {
            "notebook_env": {
                "managed": True,
                "role": "setup_code"
            }
        },
        "outputs": [],
        "source": [line + "\n" for line in blueprint["step2_code"].splitlines()]
    }
    return [cell1, cell2]


# =====================================================================
# BATCH ORCHESTRATION & CLI DISPATCH
# Directory walking engine for multi-notebook repo analysis, universal 
# manifest generation (requirements-all.txt), argument parsing, and main() execution.
# =====================================================================

class RepoEnvironmentMap:
    def __init__(self, target_dir: str) -> None:
        self.target_dir = target_dir
        self.scan_results: List[NotebookScanResult] = []
        self.non_python_files: List[NotebookScanResult] = []
        self.parse_errors: List[NotebookScanResult] = []
        self.global_imports: Set[str] = set()
        self.package_to_notebooks: Dict[str, List[Path]] = {}
        self.harvested_packages_to_notebooks: Dict[str, List[Path]] = {}
        self.url_to_notebooks: Dict[str, List[Path]] = {}
        self.local_repo_modules: Set[str] = discover_local_repo_modules(target_dir)

    def add_result(self, result: NotebookScanResult) -> None:
        if result.parse_error:
            self.parse_errors.append(result)
            return
        if not result.is_python:
            self.non_python_files.append(result)
            return

        self.scan_results.append(result)
        for imp in result.imports:
            if imp not in STD_LIB:
                self.global_imports.add(imp)
                self.package_to_notebooks.setdefault(imp, []).append(result.path)

        for pkg in result.harvested_pkgs:
            if pkg not in STD_LIB:
                self.global_imports.add(pkg)
                self.harvested_packages_to_notebooks.setdefault(pkg, []).append(result.path)

        for url in result.harvested_urls:
            self.url_to_notebooks.setdefault(url, []).append(result.path)


def select_primary_index_url(url_to_notebooks: Dict[str, List[Path]]) -> Tuple[Optional[str], Optional[str]]:
    """Deterministically selects primary index URL."""
    if not url_to_notebooks:
        return None, None

    sorted_urls = sorted(url_to_notebooks.keys())
    
    def sorting_key(url: str) -> Tuple[int, str, str]:
        notebooks = sorted([str(p) for p in url_to_notebooks[url]])
        count = len(notebooks)
        first_nb = notebooks[0] if notebooks else ""
        return (-count, first_nb, url)

    best_url = sorted(sorted_urls, key=sorting_key)[0]
    count = len(url_to_notebooks[best_url])
    total_urls = len(url_to_notebooks)
    
    if total_urls > 1:
        reason = f"Majority rule (used in {count} notebook(s); selected over {total_urls - 1} runner-up URL(s))"
    else:
        reason = f"Sole index URL harvested across batch ({count} notebook(s))"

    return best_url, reason


def walk_and_scan_directory(target_dir: str) -> RepoEnvironmentMap:
    """Recursively scans directory for .ipynb files in strict batch mode."""
    repo_map = RepoEnvironmentMap(target_dir)
    target_path = Path(target_dir)

    for root, dirs, files in os.walk(target_path):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('venv', 'env', '__pycache__')]
        for file in sorted(files):
            if file.endswith('.ipynb'):
                full_path = Path(root) / file
                success, imports, submodules, code_sources, err, lang_label, guarded_imports, dyn_warnings = extract_from_file(str(full_path), strict=True)
                writefile_imports = extract_writefile_imports_from_sources(code_sources)
                
                h_pkgs, base_urls, extra_urls, m_warns, m_notices = harvest_cell_magics_and_commands(code_sources)
                harvested_urls = base_urls.union(extra_urls)

                parse_err = err if (not success and "Skipped non-Python notebook" not in (err or "")) else None                
                res = NotebookScanResult(
                    path=full_path,
                    is_python=success,
                    lang_label=lang_label,
                    parse_error=parse_err,
                    imports=imports,
                    submodules=submodules,
                    guarded_imports=guarded_imports,
                    dynamic_warnings=dyn_warnings,
                    code_sources=code_sources,
                    harvested_urls=harvested_urls,
                    writefile_imports=writefile_imports,
                    harvested_pkgs=h_pkgs,
                    base_index_urls=base_urls,
                    extra_index_urls=extra_urls,
                    magic_warnings=m_warns,
                    magic_notices=m_notices
                )
                repo_map.add_result(res)

    return repo_map


def generate_batch_analysis_report(
    repo_map: RepoEnvironmentMap, 
    frozen_env: Dict[str, str], 
    pkg_dist_map: Dict[str, List[str]], 
    batch_hw_cache: Optional[GpuInfo]
) -> Tuple[str, bool]:
    """Generates stdout report for batch analysis mode."""
    py_count = len(repo_map.scan_results)
    non_py_count = len(repo_map.non_python_files)
    err_count = len(repo_map.parse_errors)

    out = []
    out.append("=" * 80)
    out.append("BATCH ENVIRONMENT ANALYSIS REPORT")
    out.append(f"Target Directory: {repo_map.target_dir}")
    out.append(f"Active Interpreter: {sys.executable}")
    out.append("=" * 80 + "\n")

    out.append("📁 NOTEBOOK INVENTORY & LANGUAGE SCAN:")
    out.append(f"  • Python (.ipynb): {py_count} files analyzed")
    
    if non_py_count > 0:
        lang_counts: Dict[str, int] = {}
        for item in repo_map.non_python_files:
            lang_counts[item.lang_label] = lang_counts.get(item.lang_label, 0) + 1
        lang_str = ", ".join([f"{k} ({v})" for k, v in lang_counts.items()])
        out.append(f"  • Non-Python skipped: {non_py_count} files [{lang_str}]")
    else:
        out.append("  • Non-Python skipped: 0 files")

    out.append(f"  • File / Parse Errors: {err_count} files")
    out.append("")

    if err_count > 0:
        out.append("❌ FILE & PARSE ERRORS:")
        for err_res in repo_map.parse_errors:
            out.append(f"  • {err_res.path}")
            out.append(f"    └─ Cause: {err_res.parse_error}")
        out.append("")

    matched_packages: Set[str] = set()
    missing_packages: Dict[str, List[str]] = {}
    promotions: List[str] = []
    dynamic_warnings: List[str] = []
    aggregated_magic_warnings: List[str] = []
    aggregated_magic_notices: List[str] = []

    for res in repo_map.scan_results:
        pinned_entries, notes = build_manifest_entries(
            res.imports, 
            res.submodules, 
            frozen_env, 
            pkg_dist_map, 
            guarded_imports=res.guarded_imports,
            local_repo_modules=repo_map.local_repo_modules
        )
        for note in notes:
            if note not in promotions:
                promotions.append(note)

        for warn in res.dynamic_warnings:
            if warn not in dynamic_warnings:
                dynamic_warnings.append(warn)

        for warn in res.magic_warnings:
            if warn not in aggregated_magic_warnings:
                aggregated_magic_warnings.append(warn)

        for notice in res.magic_notices:
            if notice not in aggregated_magic_notices:
                aggregated_magic_notices.append(notice)

        for pin_entry in pinned_entries:
            if pin_entry.startswith("#"):
                if "platform pseudo-module" in pin_entry or "local repo module" in pin_entry:
                    continue
                pypi_name = pin_entry.split()[1]
                missing_packages.setdefault(pypi_name, []).append(res.path.name)
            else:
                pkg_name = pin_entry.split("==")[0]
                matched_packages.add(pkg_name)

        # Process harvested magic packages
        for pkg in res.harvested_pkgs:
            if pkg in STD_LIB or pkg in PLATFORM_PSEUDO_MODULES or pkg in repo_map.local_repo_modules:
                continue
            pypi_name = IMPORT_TO_PYPI_MAP.get(pkg, pkg)
            matched_pin = frozen_env.get(pypi_name.lower())
            if matched_pin:
                pkg_name = matched_pin.split("==")[0]
                matched_packages.add(pkg_name)
            else:
                missing_packages.setdefault(pypi_name, []).append(res.path.name)

    out.append(f"📦 IMPORTED DEPENDENCY FOOTPRINT (Across {py_count} Python notebooks):")
    out.append(f"  • Installed & Matched: {len(matched_packages)} packages ({', '.join(sorted(matched_packages)[:5])}{'...' if len(matched_packages) > 5 else ''})")
    
    if missing_packages:
        out.append(f"  • Uninstalled in active env: {len(missing_packages)} package(s)")
        for pkg, nbs in sorted(missing_packages.items()):
            nb_list = ", ".join(sorted(set(nbs))[:3])
            more = f", +{len(set(nbs))-3} more" if len(set(nbs)) > 3 else ""
            out.append(f"      - {pkg} (imported in: {nb_list}{more})")
    else:
        out.append("  • Uninstalled in active env: 0 packages")
    out.append("")

    if dynamic_warnings or aggregated_magic_warnings:
        out.append("⚠️ WARNINGS DETECTED:")
        for warn in dynamic_warnings:
            out.append(f"  • {warn}")
        for warn in aggregated_magic_warnings:
            out.append(f"  • {warn}")
        out.append("")

    if aggregated_magic_notices:
        out.append("ℹ️ NOTICES:")
        for notice in aggregated_magic_notices:
            out.append(f"  • {notice}")
        out.append("")

    if promotions:
        out.append("💡 DYNAMIC PROMOTIONS DETECTED:")
        for note in promotions:
            out.append(f"  • {note}")
        out.append("")

    # Hardware & Index Audit
    out.append("⚡ HARDWARE & INDEX AUDIT:")
    if batch_hw_cache and batch_hw_cache.get("has_gpu"):
        out.append(f"  • Active Hardware Accelerator: {batch_hw_cache['device_name']}")
    else:
        out.append("  • Active Hardware Accelerator: None (CPU-only execution environment)")

    primary_url, url_reason = select_primary_index_url(repo_map.url_to_notebooks)
    if primary_url:
        out.append(f"  • Primary Index URL: {primary_url}")
        out.append(f"    └─ Selection Rule: {url_reason}")
    else:
        out.append("  • Extra Index URLs Harvested: None")

    out.append("\n" + "-" * 80)
    if err_count > 0:
        out.append("STATUS: ⚠️ ATTENTION REQUIRED - Parse errors present. Resolve file issues above.")
    else:
        out.append(f"STATUS: No blocking file errors found across {py_count} Python notebooks. Output mode (--output) can be executed.")
    out.append("=" * 80)

    return "\n".join(out), err_count == 0


def generate_universal_manifest(
    repo_map: RepoEnvironmentMap, frozen_env: Dict[str, str], pkg_dist_map: Dict[str, List[str]]
) -> str:
    """Generates content string for requirements-all.txt."""
    lines = []
    lines.append("# =====================================================================")
    lines.append("# REPOSITORY UNIVERSAL DEPENDENCY MANIFEST")
    lines.append(f"# Target Directory: {repo_map.target_dir}")
    lines.append(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    primary_url, url_reason = select_primary_index_url(repo_map.url_to_notebooks)
    if primary_url:
        lines.append("#")
        lines.append(f"# Primary Download Index: {primary_url}")
        lines.append(f"# Selection Rule: {url_reason}")

    lines.append("# =====================================================================\n")

    if repo_map.url_to_notebooks:
        for url in sorted(repo_map.url_to_notebooks.keys()):
            lines.append(f"--extra-index-url {url}")

    pinned_entries_set: Set[str] = set()
    for res in repo_map.scan_results:
        entries, _ = build_manifest_entries(
            res.imports, 
            res.submodules, 
            frozen_env, 
            pkg_dist_map, 
            guarded_imports=res.guarded_imports,
            local_repo_modules=repo_map.local_repo_modules
        )
        pinned_entries_set.update(entries)

        aux_entries = build_auxiliary_tool_entries(res.harvested_pkgs, res.imports, frozen_env)
        for aux in aux_entries:
            if not aux.startswith("\n# ---"):
                pinned_entries_set.add(aux)

    for entry in sorted(pinned_entries_set):
        lines.append(entry)

    return "\n".join(lines)


def apply_output_to_notebook(
    scan_res: NotebookScanResult, 
    frozen_env: Dict[str, str], 
    pkg_dist_map: Dict[str, List[str]], 
    batch_hw_cache: Optional[GpuInfo], 
    suffix: str = "_merged", 
    in_place: bool = False,
    local_repo_modules: Optional[Set[str]] = None
) -> Path:
    """Writes per-notebook locked file or replaces cells in-place."""
    pinned_manifest, _ = build_manifest_entries(
        scan_res.imports, 
        scan_res.submodules, 
        frozen_env, 
        pkg_dist_map, 
        guarded_imports=scan_res.guarded_imports,
        local_repo_modules=local_repo_modules
    )
    harvested_pkgs, base_urls, extra_urls, _, _ = harvest_cell_magics_and_commands(scan_res.code_sources)
    aux_entries = build_auxiliary_tool_entries(harvested_pkgs, scan_res.imports, frozen_env)
    writefile_entries = build_writefile_tool_entries(scan_res.writefile_imports, scan_res.imports, frozen_env)
    
    manifest_lines, local_tagged, _ = process_package_requirements(
        pinned_manifest, scan_res.harvested_urls, base_urls=base_urls, auxiliary_entries=aux_entries, writefile_entries=writefile_entries
    )
    
    gpu_info: Optional[GpuInfo] = None
    if batch_hw_cache:
        nb_fw = set(batch_hw_cache.get("frameworks", [])).intersection(scan_res.imports)
        if nb_fw:
            gpu_info = dict(batch_hw_cache)
            gpu_info["frameworks"] = list(nb_fw)

    blueprint = generate_production_blueprint(manifest_lines, local_tagged_info=local_tagged, gpu_info=gpu_info)
    managed_cells = create_managed_cells(blueprint)

    with open(scan_res.path, 'r', encoding='utf-8') as f:
        nb_data = json.load(f)

    cells = nb_data.get("cells", [])

    if in_place:
        target_path = scan_res.path
        non_managed_cells = [
            c for c in cells 
            if not (isinstance(c.get("metadata"), dict) and c.get("metadata", {}).get("notebook_env", {}).get("managed") is True)
        ]
        nb_data["cells"] = managed_cells + non_managed_cells
    else:
        stem = scan_res.path.stem
        target_path = scan_res.path.parent / f"{stem}{suffix}.ipynb"
        nb_data["cells"] = managed_cells + cells

    with open(target_path, 'w', encoding='utf-8') as f:
        json.dump(nb_data, f, indent=1)

    return target_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate environment lockfiles for Jupyter Notebooks.")
    parser.add_argument("notebook", nargs="?", help="Path to target .ipynb file or directory (when using --batch).")
    parser.add_argument("--full-freeze", action="store_true", help="Append full environment pip freeze after targeted manifest.")
    parser.add_argument("--quiet", action="store_true", help="Suppress diagnostic and status logging outputs.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose debug output.")
    
    # Batch Flags
    parser.add_argument("--batch", metavar="DIR", help="Run in batch mode across all notebooks in specified directory.")
    parser.add_argument("--analyze", action="store_true", help="Run batch analysis mode (default when --batch is provided).")
    parser.add_argument("--universal", action="store_true", help="Generate root requirements-all.txt universal manifest.")
    parser.add_argument("--output", action="store_true", help="Generate per-notebook merged lockfiles.")
    parser.add_argument("--suffix", default="_merged", help="File suffix for merged notebook outputs (default: '_merged').")
    parser.add_argument("--in-place", action="store_true", help="Overwrite original notebooks in-place instead of creating companion files.")

    args, unknown = parser.parse_known_args()

    if args.quiet:
        logger.setLevel(logging.ERROR)
    elif args.verbose:
        logger.setLevel(logging.DEBUG)

    # Environment inspection
    frozen_env, raw_full_freeze = get_installed_environment()
    pkg_dist_map = importlib.metadata.packages_distributions() if hasattr(importlib.metadata, "packages_distributions") else {}
    batch_hw_cache = inspect_gpu_environment({"torch", "tensorflow", "jax"})

    # --- BATCH DISPATCH ---
    target_batch_dir = args.batch or (args.notebook if args.notebook and os.path.isdir(args.notebook) else None)

    if target_batch_dir:
        repo_map = walk_and_scan_directory(target_batch_dir)
        report_text, is_clean = generate_batch_analysis_report(repo_map, frozen_env, pkg_dist_map, batch_hw_cache)
        print(report_text)

        if not is_clean and (args.universal or args.output):
            logger.error("\n❌ Execution aborted: Resolve file/parse errors before running --universal or --output.")
            sys.exit(1)

        if args.universal:
            uni_content = generate_universal_manifest(repo_map, frozen_env, pkg_dist_map)
            out_file = Path(target_batch_dir) / "requirements-all.txt"
            with open(out_file, 'w', encoding='utf-8') as f:
                f.write(uni_content)
            logger.info(f"\n✅ Wrote universal repository manifest to '{out_file}'")

        if args.output:
            logger.info(f"\n🚀 Writing per-notebook locked files ({'in-place' if args.in_place else 'suffix: ' + args.suffix})...")
            for res in repo_map.scan_results:
                written_path = apply_output_to_notebook(
                    res, 
                    frozen_env, 
                    pkg_dist_map, 
                    batch_hw_cache, 
                    suffix=args.suffix, 
                    in_place=args.in_place,
                    local_repo_modules=repo_map.local_repo_modules
                )
                logger.info(f"  • Updated '{written_path.name}'")
            logger.info("✅ Batch output complete.")

        sys.exit(0)

    # --- SINGLE FILE / LIVE SESSION DISPATCH ---
    in_live_ipython = False
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            in_live_ipython = True
    except ImportError:
        pass

    target_single_file_dir = str(Path(args.notebook).parent) if (args.notebook and not os.path.isdir(args.notebook)) else "."
    single_file_local_modules = discover_local_repo_modules(target_single_file_dir)

    if args.notebook and not os.path.isdir(args.notebook):
        logger.info(f"🔍 [Path A] Analyzing saved notebook file '{args.notebook}' via AST...")
        logger.info(f"📌 Active Python Interpreter: {sys.executable}\n")
        
        success, imports, submodules, code_sources, error_msg, _, guarded_imports, dyn_warnings = extract_from_file(args.notebook, strict=False)
        if not success:
            logger.error(f"❌ Error: {error_msg}")
            sys.exit(1)
            
        writefile_imports = extract_writefile_imports_from_sources(code_sources)
    elif in_live_ipython:
        logger.info("🔍 [Path B] Analyzing live IPython session kernel history via AST...")
        imports, submodules, code_sources, guarded_imports, dyn_warnings = extract_from_active_session()
        writefile_imports = extract_writefile_imports_from_sources(code_sources)
    else:
        parser.print_help()
        sys.exit(1)

    harvested_pkgs, base_urls, extra_urls, magic_warns, magic_notices = harvest_cell_magics_and_commands(code_sources)
    harvested_urls = extra_urls.union(base_urls)
    gpu_info = inspect_gpu_environment(imports)
    
    # Combined warnings
    all_warnings = dyn_warnings + magic_warns

    # Unified manifest building
    pinned_manifest, promotion_notices = build_manifest_entries(
        imports, 
        submodules, 
        frozen_env, 
        pkg_dist_map, 
        guarded_imports=guarded_imports,
        local_repo_modules=single_file_local_modules
    )
    
    # Build auxiliary tool and writefile dependency sections
    aux_entries = build_auxiliary_tool_entries(harvested_pkgs, imports, frozen_env)
    writefile_entries = build_writefile_tool_entries(writefile_imports, imports, frozen_env)

    manifest_lines, local_tagged_info, warnings = process_package_requirements(
        pinned_manifest, harvested_urls, base_urls=base_urls, auxiliary_entries=aux_entries, writefile_entries=writefile_entries
    )
    full_freeze_lines = raw_full_freeze if args.full_freeze else None

    # DIAGNOSTIC LOGGING (To stderr via logger)
    if warnings:
        logger.warning("⚠️ HARDWARE BUILD WARNINGS:")
        for pkg in warnings:
            logger.warning(f"  • Specific hardware build detected: `{pkg}`")
            logger.warning("    No matching download URL was found in code cells. If installation fails on target machines, ensure runtime matches or supply an --extra-index-url.\n")

    if all_warnings:
        for warn in all_warnings:
            logger.warning(f"{warn}")
        logger.warning("")

    if magic_notices:
        for notice in magic_notices:
            logger.info(f"{notice}")
        logger.info("")

    if gpu_info:
        if gpu_info.get("has_gpu"):
            logger.info(f"⚡ Active accelerator detected: {gpu_info['device_name']}\n")
        elif gpu_info.get("frameworks"):
            fw_list = ", ".join(gpu_info["frameworks"])
            logger.warning(f"⚠️ Acceleration Framework ({fw_list}) imported, but NO active accelerator detected in host runtime.\n")

    if promotion_notices:
        for note in promotion_notices:
            logger.info(note)
        logger.info("")

    blueprint = generate_production_blueprint(
        manifest_lines, 
        full_freeze_lines=full_freeze_lines, 
        local_tagged_info=local_tagged_info,
        gpu_info=gpu_info
    )

    # DELIVERABLE OUTPUT (To stdout via print)
    print("--- [ STEP 1: PASTE INTO CELL 1 (MARKDOWN) ] ---\n")
    print(blueprint["step1_markdown"])
    print("\n" + "="*80 + "\n")

    print("--- [ STEP 2: PASTE INTO CELL 2 (CODE) ] ---\n")
    print(blueprint["step2_code"])
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
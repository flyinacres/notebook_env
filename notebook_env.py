#!/usr/bin/env python3
"""
notebook_env.py (v37)
Headless Jupyter Notebook Dependency Scanner & Lockfile Generator.

Standalone, zero-dependency utility for analyzing notebook environments,
detecting GPU/accelerator requirements, harvesting index URLs, and emitting
reproducible lockfile manifests (`pinned_requirements.txt`).

=====================================================================
🚀 QUICKSTART FOR JUPYTER / COLAB / DATABRICKS USERS
=====================================================================
If you are running inside a Jupyter notebook cell:
  1. Paste this entire file into a notebook cell.
  2. Run:
       import notebook_env as ne
       ne.main()
  3. Copy the output setup cells into the top of your notebook.

For full CLI documentation, batch directory workflows, and detailed instructions, 
see the repository README:
👉 https://github.com/flyinacres/notebook_env/blob/main/README.md

Execution Modes:
  1. Single Notebook CLI:  python notebook_env.py notebook.ipynb [--output | --in-place]
  2. Batch Repo Directory: python notebook_env.py --batch ./repo [--universal [FILENAME]] [--output | --in-place]
  3. Live IPython Kernel:   import notebook_env as ne; ne.main()
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
import functools
import logging
import warnings
import subprocess
import importlib.metadata
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Set, Dict, List, Tuple, Optional, Any, TypedDict, Callable, NamedTuple


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


# =====================================================================
# SYSTEM CONSTANTS & CONFIGURATION DEFAULTS
# Configurable manifest filenames, documentation URLs, directory ignores, and hardware targets.
# =====================================================================

DEFAULT_PINNED_MANIFEST_NAME: str = "pinned_requirements.txt"
DEFAULT_UNIVERSAL_MANIFEST_NAME: str = "requirements-all.txt"

HELP_URL: str = "https://github.com/flyinacres/notebook_env/blob/main/HELP.md"
README_URL: str = "https://github.com/flyinacres/notebook_env/blob/main/README.md"

DEFAULT_IGNORED_DIRS: Set[str] = {
    ".git", ".venv", "venv", "env", "__pycache__", ".ipynb_checkpoints", "build", "dist"
}

SUPPORTED_GPU_FRAMEWORKS: Set[str] = {"torch", "tensorflow", "jax"}

CANONICAL_TO_FRAMEWORK_DISPLAY: Dict[str, str] = {
    "torch": "PyTorch",
    "tensorflow": "TensorFlow",
    "jax": "JAX"
}


class StatusLabel:
    """Standardized metadata and language classification status labels."""
    PYTHON = "python"
    CORRUPTED = "corrupted"
    ERROR = "error"
    UNKNOWN = "unknown"
    MISSING_METADATA = "missing metadata"


@dataclass
class GpuInfo:
    """
    Payload representing active host accelerator capabilities across PyTorch, TensorFlow, and JAX.

    Fields:
        has_gpu: True if at least one framework verified an active GPU/accelerator.
        type: Primary hardware string (e.g., 'NVIDIA CUDA', 'Apple Silicon MPS', 'GPU', 'TPU').
        active_framework: Human-readable display label of the primary active framework (e.g., 'PyTorch').
        device_name: Primary device hardware descriptor string.
        frameworks: List of canonical framework stems detected in imports (['torch', 'tensorflow', 'jax']).
        framework_devices: Map of canonical framework stem -> verified device descriptor string (or None if CPU-only).
    """
    has_gpu: bool = False
    type: Optional[str] = None
    active_framework: Optional[str] = None
    device_name: Optional[str] = None
    frameworks: List[str] = field(default_factory=list)
    framework_devices: Dict[str, Optional[str]] = field(default_factory=dict)


class BlueprintResult(TypedDict):
    """Cell blueprint output strings for Cell 1 (Markdown) and Cell 2 (Python script)."""
    step1_markdown: str
    step2_code: str


@dataclass
class NotebookScanResult:
    """
    Complete AST and metadata analysis payload for an individual notebook file.

    Fields:
        path: Absolute or relative Path object pointing to the notebook file.
        is_python: True if notebook kernelspec / metadata indicates Python language.
        lang_label: String label of the detected kernel language (e.g. 'python', 'R').
        parse_error: Error message string if JSON or AST parsing failed.
        imports: Top-level imported package stems (e.g., 'torch', 'pandas').
        submodules: Map of top-level package -> set of submodules imported (e.g. {'matplotlib': {'matplotlib.pyplot'}}).
        guarded_imports: Imports occurring exclusively inside try/except or conditional blocks.
        dynamic_warnings: Warnings triggered by non-literal dynamic import calls.
        code_sources: Raw code cell string bodies extracted from notebook cells.
        harvested_urls: Combined set of harvested base and extra index download URLs.
            Defaults to None (not str's empty-set) so __post_init__ can distinguish
            "caller hasn't computed this yet, please derive it" from "caller already
            ran the harvest and confirmed there are none" — an empty set is a
            perfectly valid, common result (most notebooks reference no custom
            index), and re-deriving it in that case would silently repeat a full
            harvest_cell_magics_and_commands() pass for no benefit. Always a concrete
            Set[str] (never None) once __post_init__ has run.
        writefile_imports: Imports occurring exclusively inside %%writefile generated scripts.
        harvested_pkgs: Packages installed via cell magics (%pip / !pip) but not imported in Python code.
        base_index_urls: Base index URLs harvested via --index-url / -i.
        extra_index_urls: Supplemental index URLs harvested via --extra-index-url.
        magic_warnings: Warnings for unresolvable magics (e.g., -r requirements.txt references).
        magic_notices: Informational notices for conda / apt-get calls outside pip manifests.
    """
    path: Path
    is_python: bool
    lang_label: str
    parse_error: Optional[str] = None
    imports: Set[str] = field(default_factory=set)
    submodules: Dict[str, Set[str]] = field(default_factory=dict)
    guarded_imports: Set[str] = field(default_factory=set)
    dynamic_warnings: List[str] = field(default_factory=list)
    code_sources: List[str] = field(default_factory=list)
    harvested_urls: Optional[Set[str]] = None
    writefile_imports: Set[str] = field(default_factory=set)
    harvested_pkgs: Set[str] = field(default_factory=set)
    base_index_urls: Set[str] = field(default_factory=set)
    extra_index_urls: Set[str] = field(default_factory=set)
    magic_warnings: List[str] = field(default_factory=list)
    magic_notices: List[str] = field(default_factory=list)

    def __post_init__(self):
        # Only auto-derive when harvested_urls was never supplied at all (None).
        # An explicitly-passed empty set means the caller already harvested and
        # confirmed there are no index URLs — that's the common case and must
        # NOT trigger a second full harvest pass. See field docstring above.
        if self.harvested_urls is None:
            if self.code_sources:
                logger.debug(
                    f"NotebookScanResult for '{self.path.name}' constructed without "
                    "harvested_urls; auto-deriving via a fresh harvest_cell_magics_and_commands() "
                    "pass. Pass harvested_urls explicitly (even if empty) to skip this."
                )
                self.harvested_urls = harvest_index_urls_from_sources(self.code_sources)
            else:
                self.harvested_urls = set()


@dataclass
class ExtractionResult:
    """
    Encapsulates the raw extraction payload from reading a notebook file.

    Fields:
        success: True if the file was read and parsed successfully as Python.
        lang_label: Language metadata label detected in kernelspec/language_info.
        imports: Top-level imported package stems.
        submodules: Map of top-level package -> set of imported submodules.
        code_sources: Raw code cell string bodies.
        error_msg: Diagnostic error message if reading/parsing failed.
        guarded_imports: Imports occurring inside conditional or try/except blocks.
        dynamic_warnings: Warnings for non-literal dynamic import calls.
    """
    success: bool
    lang_label: str
    imports: Set[str] = field(default_factory=set)
    submodules: Dict[str, Set[str]] = field(default_factory=dict)
    code_sources: List[str] = field(default_factory=list)
    error_msg: Optional[str] = None
    guarded_imports: Set[str] = field(default_factory=set)
    dynamic_warnings: List[str] = field(default_factory=list)

    def __iter__(self):
        """Legacy tuple-unpacking fallback for backward compatibility."""
        return iter((
            self.success,
            self.imports,
            self.submodules,
            self.code_sources,
            self.error_msg,
            self.lang_label,
            self.guarded_imports,
            self.dynamic_warnings,
        ))


@dataclass
class HarvestResult:
    """
    Encapsulates harvested packages, index URLs, warnings, and notices from cell magics.

    Fields:
        harvested_packages: Auxiliary CLI tool packages installed via %pip / !pip.
        base_index_urls: Base download index URLs harvested via --index-url or -i.
        extra_index_urls: Extra download index URLs harvested via --extra-index-url.
        magic_warnings: Warnings for unresolvable magics (-r requirements.txt references).
        magic_notices: Informational notices for non-pip calls (conda, apt-get).
    """
    harvested_packages: Set[str] = field(default_factory=set)
    base_index_urls: Set[str] = field(default_factory=set)
    extra_index_urls: Set[str] = field(default_factory=set)
    magic_warnings: List[str] = field(default_factory=list)
    magic_notices: List[str] = field(default_factory=list)

    def __iter__(self):
        """Legacy tuple-unpacking fallback for backward compatibility."""
        return iter((
            self.harvested_packages,
            self.base_index_urls,
            self.extra_index_urls,
            self.magic_warnings,
            self.magic_notices,
        ))


@dataclass
class BatchAnalysisSummary:
    """
    Aggregated analysis metrics across all notebooks in a batch repo scan.

    Fields:
        target_dir: Target directory path evaluated during the batch scan.
        total_python_notebooks: Count of successfully parsed Python notebooks.
        non_python_count: Count of non-Python notebooks skipped.
        non_python_languages: Map of language name -> file count for skipped files.
        parse_errors: List of NotebookScanResult objects for corrupted/unparseable files.
        matched_packages: Set of PyPI package names successfully matched to active env.
        missing_packages: Map of missing package name -> list of importing notebook names.
        promotions: List of dynamic extra promotion notices generated across the batch.
        dynamic_warnings: List of dynamic import warnings harvested across the batch.
        magic_warnings: List of cell magic warnings harvested across the batch.
        magic_notices: List of cell magic notices harvested across the batch.
        batch_hardware_warnings: Map of package name with local tag -> list of notebook names missing index URLs.
        primary_url: Deterministically selected primary download index URL for repo.
        primary_url_reason: Selection rule explanation for primary index URL selection.
        batch_hw_cache: Host GPU/accelerator inspection cache object.
    """
    target_dir: str
    total_python_notebooks: int = 0
    non_python_count: int = 0
    non_python_languages: Dict[str, int] = field(default_factory=dict)
    parse_errors: List[NotebookScanResult] = field(default_factory=list)
    matched_packages: Set[str] = field(default_factory=set)
    missing_packages: Dict[str, List[str]] = field(default_factory=dict)
    promotions: List[str] = field(default_factory=list)
    dynamic_warnings: List[str] = field(default_factory=list)
    magic_warnings: List[str] = field(default_factory=list)
    magic_notices: List[str] = field(default_factory=list)
    batch_hardware_warnings: Dict[str, List[str]] = field(default_factory=dict)
    primary_url: Optional[str] = None
    primary_url_reason: Optional[str] = None
    batch_hw_cache: Optional[GpuInfo] = None

    @property
    def is_clean(self) -> bool:
        """Returns True if no blocking parse/file errors exist in the batch."""
        return len(self.parse_errors) == 0


# =====================================================================
# MAPPINGS, PLATFORM INJECTIONS & STDLIB LOOKUP
# Data structures for import-to-PyPI resolution, platform pseudo-modules,
# transitive framework relationships, and Python standard library filtering.
# =====================================================================

# Maps Python import top-level stems (e.g. `import cv2`) to their standard PyPI package names (`opencv-python`).
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

# Injected cloud platform modules provided natively by runtimes (Databricks, Colab, Kaggle).
# These modules have no PyPI equivalent and are excluded from uninstalled-package warnings.
PLATFORM_PSEUDO_MODULES: Set[str] = {
    "dbutils",
    "kaggle_secrets",
    "google.colab",
    "pyspark.dbutils"
}

# Maps high-level framework wrappers to their core underlying GPU acceleration framework.
# (e.g., importing `fastai` requires `torch` acceleration checks).
TRANSITIVE_FRAMEWORK_MAP: Dict[str, str] = {
    "fastai": "torch",
    "torchvision": "torch",
    "torchaudio": "torch",
    "timm": "torch",
    "keras": "tensorflow",
    "flax": "jax",
}

# Inverts CANONICAL_TO_FRAMEWORK_DISPLAY for input mapping.
FRAMEWORK_NAME_TO_CANONICAL: Dict[str, str] = {
    v: k for k, v in CANONICAL_TO_FRAMEWORK_DISPLAY.items()
}

# Python standard library module stems. Used to filter built-in modules out of PyPI lockfile manifests.
STD_LIB: Set[str] = set(sys.stdlib_module_names) if hasattr(sys, 'stdlib_module_names') else {
    "os", "sys", "re", "json", "ast", "subprocess", "datetime", "math", "random", 
    "time", "pathlib", "typing", "collections", "itertools", "functools", "shutil"
}


def discover_local_repo_modules(target_dir: str) -> Set[str]:
    """Scans target_dir for valid top-level Python modules and packages to prevent false-positive PyPI warnings."""
    local_mods: Set[str] = set()
    target_path = Path(target_dir)
    if not target_path.exists():
        return local_mods

    try:
        # 1. Top-level .py files (directly importable as `import foo`)
        for entry in target_path.iterdir():
            if entry.is_file() and entry.suffix == ".py" and entry.stem != "__init__":
                local_mods.add(entry.stem)
            elif entry.is_dir() and entry.name not in DEFAULT_IGNORED_DIRS and not entry.name.startswith('.'):
                # 2. Top-level packages/directories (e.g. `import src` or `from src.utils import x`)
                # Only register if it contains Python files
                if any(entry.rglob("*.py")):
                    local_mods.add(entry.name)
    except Exception:
        pass

    return local_mods


def _memoize_for_run(func: Callable) -> Callable:
    """
    Memoizes a function whose arguments may include mutable Set/Dict values
    that functools.lru_cache can't hash directly (it hashes raw arguments,
    and sets/dicts aren't hashable).

    Cache-key handling:
      - dict arguments are keyed by id(), not content. Within a single run,
        the same dict object (e.g. frozen_env, pkg_dist_map, or a given
        notebook's own res.submodules) is passed by reference to every call
        site that needs it, so identity is a correct and far cheaper stand-in
        than re-hashing potentially hundreds of entries (e.g. a full `pip
        freeze` map) on every call. This assumes callers don't rebuild an
        equal-but-distinct dict between calls for what should be a cache hit;
        that holds for every current call site in this file.
      - set arguments are converted to frozenset for hashing (small,
        per-notebook sets — cheap to convert).
      - Path/str/None/etc. pass through unchanged; already hashable.

    Returned Set/List/tuple-of-those values are shallow-copied on every call
    so callers never receive a shared reference to the cached object — an
    in-place mutation by one caller (e.g. `.add(...)`) can't corrupt the
    value seen by the next caller.

    IMPORTANT — cache lifetime: this tool can run repeatedly inside a single
    long-lived process (a live IPython kernel — see "Live IPython Kernel" in
    the module docstring: `import notebook_env as ne; ne.main()`), where the
    notebook's own files may legitimately change between calls to main().
    An uncleared cache would silently keep returning pre-change results.
    main() unconditionally clears every memoized function's cache via
    .cache_clear() at the top of every call, so caching is scoped to a
    single logical run and never crosses runs. Do not rely on this decorator
    for correctness without that call.
    """
    cache: Dict[Tuple[Any, ...], Any] = {}

    def _cache_key_part(value: Any) -> Any:
        if isinstance(value, dict):
            return id(value)
        if isinstance(value, set):
            return frozenset(value)
        return value

    def _defensive_copy(value: Any) -> Any:
        if isinstance(value, set):
            return set(value)
        if isinstance(value, list):
            return list(value)
        if isinstance(value, tuple):
            return tuple(_defensive_copy(item) for item in value)
        return value

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        key = (
            tuple(_cache_key_part(a) for a in args),
            tuple(sorted((k, _cache_key_part(v)) for k, v in kwargs.items()))
        )
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return _defensive_copy(cache[key])

    wrapper.cache_clear = cache.clear
    return wrapper


@_memoize_for_run
def get_notebook_local_modules(notebook_path: Path, root_dir: Optional[str] = None) -> Set[str]:
    """Discovers local repo modules scoped to both the notebook's immediate parent directory and repository root."""
    local_mods = discover_local_repo_modules(str(notebook_path.parent))
    if root_dir and Path(root_dir).exists():
        local_mods.update(discover_local_repo_modules(root_dir))
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
) -> ExtractionResult:
    """
    Reads a Jupyter Notebook JSON file and extracts code sources, imports, guarded state, and dynamic warnings.

    Args:
        notebook_path: Path string to the target .ipynb file.
        strict: If True, rejects notebooks with missing language metadata (used in batch mode).

    Returns:
        ExtractionResult object containing extraction status, code sources, imports, and metadata.
    """
    if not os.path.exists(notebook_path):
        return ExtractionResult(
            success=False,
            lang_label=StatusLabel.UNKNOWN,
            error_msg=f"File '{notebook_path}' not found."
        )

    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb_data = json.load(f)
    except json.JSONDecodeError:
        return ExtractionResult(
            success=False,
            lang_label=StatusLabel.CORRUPTED,
            error_msg="File is not valid JSON. Ensure the file was not truncated or saved mid-write."
        )
    except Exception as e:
        return ExtractionResult(
            success=False,
            lang_label=StatusLabel.ERROR,
            error_msg=f"Unable to read file ({type(e).__name__}). Check file permissions and path location."
        )

    if not isinstance(nb_data, dict) or "cells" not in nb_data or not isinstance(nb_data.get("cells"), list):
        return ExtractionResult(
            success=False,
            lang_label=StatusLabel.CORRUPTED,
            error_msg="Unparseable notebook structure (Missing or invalid 'cells' array)"
        )

    is_py, lang_label = detect_notebook_language(nb_data, strict=strict)
    if not is_py:
        return ExtractionResult(
            success=False,
            lang_label=lang_label,
            error_msg=f"Skipped non-Python notebook (Language: {lang_label})"
        )

    cells = nb_data.get("cells", [])
    code_sources = ["".join(c.get("source", [])) for c in cells if c.get("cell_type") == "code"]
    imports, submodules, guarded_imports, dyn_warnings = extract_imports_from_sources(code_sources)

    return ExtractionResult(
        success=True,
        lang_label=lang_label,
        imports=imports,
        submodules=submodules,
        code_sources=code_sources,
        guarded_imports=guarded_imports,
        dynamic_warnings=dyn_warnings
    )


def extract_from_active_session() -> Tuple[Set[str], Dict[str, Set[str]], List[str], Set[str], List[str]]:
    """Path B (Live Kernel): Reads IPython execution history."""
    import __main__
    code_sources = [src for src in getattr(__main__, 'In', []) if src and isinstance(src, str)]
    imports, submodules, code_sources, guarded_imports, dyn_warnings = extract_from_active_session_internal(code_sources)
    return imports, submodules, code_sources, guarded_imports, dyn_warnings


def extract_from_active_session_internal(code_sources: List[str]) -> Tuple[Set[str], Dict[str, Set[str]], List[str], Set[str], List[str]]:
    """Internal helper to parse IPython history lists via extract_imports_from_sources."""
    imports, submodules, guarded_imports, dyn_warnings = extract_imports_from_sources(code_sources)
    return imports, submodules, code_sources, guarded_imports, dyn_warnings


# =====================================================================
# AST VISITOR & DYNAMIC IMPORT PARSER
# Python AST traversal engine. Inspects import statements, submodules,
# guarded try/except blocks, and importlib/literal dynamic import calls.
# =====================================================================

class NotebookImportVisitor(ast.NodeVisitor):
    """AST visitor traversing Python code to record imports, guarded states, and dynamic calls."""
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
                    f"⚠️ Dynamic import detected via variable '{expr_repr}'. Check that this package is installed if execution fails."
                )

        self.generic_visit(node)


def extract_imports_from_sources(
    code_sources: List[str]
) -> Tuple[Set[str], Dict[str, Set[str]], Set[str], List[str]]:
    """Executes AST traversal over code sources, stripping IPython line magics."""
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

    # Subtract writefile-only imports from primary imports
    primary_imports = visitor.imports - visitor.writefile_imports

    return (
        primary_imports, 
        visitor.submodules, 
        visitor.guarded_imports, 
        visitor.dynamic_import_warnings
    )


def extract_writefile_imports_from_sources(code_sources: List[str]) -> Set[str]:
    """Dedicated helper to extract writefile imports without altering extract_imports_from_sources signature."""
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

# Standalone pip boolean flags that do not take arguments.
PIP_SINGLE_FLAGS: Set[str] = {
    "-u", "--upgrade", "-q", "--quiet", "--user", "--no-cache-dir",
    "--force-reinstall", "--no-deps", "--pre", "--break-system-packages"
}

# Pip flags expecting an immediate value parameter token.
PIP_VALUE_FLAGS: Set[str] = {
    "--extra-index-url", "--index-url", "-i", "-f", "--find-links", 
    "-t", "--target", "-e", "--editable", "-r", "--requirement"
}

# Non-Python IPython cell magic headers that mark shell script cells.
SHELL_CELL_MAGICS: Set[str] = {
    "%%bash", "%%sh", "%%zsh", "%%script", "%%cmd", "%%powershell"
}

# --- COMPILED REGEX PATTERNS ---

# Matches `--extra-index-url <url>` in pip install magic calls. Capture group 1 extracts the URL string.
EXTRA_INDEX_PATTERN = re.compile(r'--extra-index-url\s+([^\s]+)')

# Matches `--index-url <url>` or `-i <url>` base index overrides. Capture group 1 extracts the URL string.
BASE_INDEX_PATTERN = re.compile(r'(?:--index-url|-i)\s+([^\s]+)')

# Splits chained shell commands separated by ;, &&, ||, or |.
SHELL_SPLIT_PATTERN = re.compile(r'\s*(?:&&|;|\||\|\|)\s*')

# Matches `%pip install`, `!pip install`, or `pip install` commands. Capture group 1 extracts arguments.
PIP_INSTALL_PATTERN = re.compile(r'^\s*(?:%pip|!pip|pip3?)\s+install\s+(.+)$')

# Matches system package manager calls (`apt-get install`, `brew install`, `yum install`).
SYSTEM_PKG_PATTERN = re.compile(r'^\s*(?:!|%%bash|%%sh)?\s*(?:apt-get|brew|yum)\s+install\s+(.+)$')

# Matches `%conda install` or `!conda install` calls.
CONDA_INSTALL_PATTERN = re.compile(r'^\s*(?:%conda|!conda|conda)\s+install\s+(.+)$')

# Prefixes signaling local path wheels or direct VCS repository installs (e.g., git+https://...).
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
    """Scans code sources for index URLs and returns a combined set of all harvested URLs."""
    h_res = harvest_cell_magics_and_commands(code_sources)
    return h_res.base_index_urls.union(h_res.extra_index_urls)


def harvest_cell_magics_and_commands(
    code_sources: List[str]
) -> HarvestResult:
    """
    Scans code sources for cell magics, index URLs, auxiliary tools, and shell commands.

    Args:
        code_sources: List of code cell body strings.

    Returns:
        HarvestResult containing harvested packages, index URLs, warnings, and notices.
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

            for match in EXTRA_INDEX_PATTERN.finditer(clean_line):
                extra_index_urls.add(match.group(1).strip("'\""))
            
            for match in BASE_INDEX_PATTERN.finditer(clean_line):
                full_match_str = match.group(0)
                if not full_match_str.startswith("--extra-index-url"):
                    base_index_urls.add(match.group(1).strip("'\""))

            command_segments = SHELL_SPLIT_PATTERN.split(clean_line)

            for segment in command_segments:
                seg = segment.strip()
                if not seg:
                    continue

                if SYSTEM_PKG_PATTERN.match(seg):
                    magic_notices.append(
                        f"ℹ️ Cell {cell_idx} uses a system install command ('{seg}'). Note: System dependencies must be run manually by readers."
                    )
                    continue

                if CONDA_INSTALL_PATTERN.match(seg):
                    magic_notices.append(
                        f"ℹ️ Cell {cell_idx} uses 'conda install'. Conda packages are not tracked in pip requirements manifests."
                    )
                    continue

                pip_match = PIP_INSTALL_PATTERN.match(seg)
                if pip_match:
                    args_str = pip_match.group(1)

                    if "-r " in args_str or "--requirement" in args_str:
                        magic_warnings.append(
                            f"⚠️ Cell {cell_idx} references an external requirements file ('{seg}'). Ensure that file is shared alongside your notebook."
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

    return HarvestResult(
        harvested_packages=harvested_packages,
        base_index_urls=base_index_urls,
        extra_index_urls=extra_index_urls,
        magic_warnings=magic_warnings,
        magic_notices=magic_notices
    )


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
    """Builds commented-out manifest lines for CLI tools installed via cell magics not imported directly."""
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
            aux_entries.append(f"# {matched_pin}  (installed via cell command; not directly imported in Python code)")
        else:
            aux_entries.append(f"# {tool}  (installed via cell command; not found in active env)")

    return aux_entries


def build_writefile_tool_entries(
    writefile_imports: Set[str],
    primary_imports: Set[str],
    frozen_env: Dict[str, str]
) -> List[str]:
    """Builds commented-out manifest lines for dependencies imported exclusively inside %%writefile generated scripts."""
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
        return f"# {imp} (provided automatically by platform like Colab/Databricks; no install needed)", None

    if local_repo_modules and imp in local_repo_modules:
        return f"# {imp} (local folder/file next to notebook; ensure sibling files were shared)", None

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
            return f"# {matched_pin} (optional or conditional dependency inside try/except block)", None
        return f"# {pypi_name} (optional or conditional dependency inside try/except block)", None

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


@_memoize_for_run
def build_manifest_entries(
    imports: Set[str], 
    submodules: Dict[str, Set[str]], 
    frozen_env: Dict[str, str], 
    pkg_dist_map: Optional[Dict[str, List[str]]] = None,
    guarded_imports: Optional[Set[str]] = None,
    local_repo_modules: Optional[Set[str]] = None
) -> Tuple[List[str], List[str]]:
    """
    Single shared helper for generating correlated pinned manifest entries.

    Memoized via _memoize_for_run (see that function's docstring for cache
    semantics) — analyze_batch_repository, generate_universal_manifest, and
    apply_output_to_notebook / run_batch_pipeline's output loop all call this
    with identical arguments for the same notebook when --batch, --universal,
    and --output/--in-place are combined; without memoization, that recomputed
    the same manifest (including a per-import importlib.metadata.distribution()
    lookup for extras promotion) up to three times per notebook.
    """
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
    """Determines the appropriate OpenCV package variant installed in the active environment."""
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
    """Runs pip freeze to get precise version snapshots of the active runtime."""
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
    """Correlates pinned packages with harvested base/extra index URLs, auxiliary tool entries, and writefile dependencies."""
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

def expand_transitive_frameworks(imports: Set[str]) -> Set[str]:
    """
    Expands a set of import stems to include their base GPU framework.

    Checks the static TRANSITIVE_FRAMEWORK_MAP first (e.g. `fastai` -> `torch`),
    then falls back to a dynamic `importlib.metadata.requires()` lookup for
    packages not in the static map that declare a framework dependency.
    Shared by inspect_gpu_environment (host-level probing) and
    apply_output_to_notebook (per-notebook attribution) so the two stay in sync.
    """
    expanded = set(imports)
    for pkg in imports:
        base_fw = TRANSITIVE_FRAMEWORK_MAP.get(pkg)
        if base_fw:
            expanded.add(base_fw)
        else:
            try:
                reqs = importlib.metadata.requires(pkg) or []
                for req in reqs:
                    req_lower = req.lower()
                    for fw in SUPPORTED_GPU_FRAMEWORKS:
                        if fw in req_lower:
                            expanded.add(fw)
            except Exception:
                pass
    return expanded


class GpuProbeResult(NamedTuple):
    """Result of a single framework's GPU/accelerator probe."""
    accelerator_type: str
    device_name: str


def probe_torch_gpu() -> Optional[GpuProbeResult]:
    """
    Probes PyTorch for CUDA or Apple Silicon MPS acceleration.

    Returns a GpuProbeResult, or None if torch isn't installed or no
    accelerator is available. If torch is installed but the probe itself
    fails unexpectedly (not just "not installed"), logs at debug level
    (visible via --verbose) rather than silently reporting no GPU.
    """
    try:
        import torch
    except ImportError:
        return None
    try:
        if torch.cuda.is_available():
            return GpuProbeResult("NVIDIA CUDA", f"{torch.cuda.get_device_name(0)} (via PyTorch)")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return GpuProbeResult("Apple Silicon MPS", "Apple Silicon GPU (Metal via PyTorch)")
    except Exception as e:
        logger.debug(f"PyTorch GPU probe failed unexpectedly: {e}")
    return None


def probe_tensorflow_gpu() -> Optional[GpuProbeResult]:
    """
    Probes TensorFlow for GPU acceleration.

    Returns a GpuProbeResult, or None if tensorflow isn't installed or no
    GPU is available. If tensorflow is installed but the probe itself fails
    unexpectedly, logs at debug level rather than silently reporting no GPU.
    """
    try:
        import tensorflow as tf
    except ImportError:
        return None
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            return None
        dev_name = "NVIDIA GPU (via TensorFlow)"
        try:
            details = tf.config.experimental.get_device_details(gpus[0])
            dev_name = f"{details.get('device_name', 'NVIDIA GPU')} (via TensorFlow)"
        except Exception:
            pass  # Cosmetic detail lookup only; falls back to the generic name above.
        return GpuProbeResult("GPU", dev_name)
    except Exception as e:
        logger.debug(f"TensorFlow GPU probe failed unexpectedly: {e}")
    return None


def probe_jax_gpu() -> Optional[GpuProbeResult]:
    """
    Probes JAX for GPU/TPU acceleration.

    Returns a GpuProbeResult, or None if jax isn't installed or no
    accelerator is available. If jax is installed but the probe itself
    fails unexpectedly, logs at debug level rather than silently reporting no GPU.
    """
    try:
        import jax
    except ImportError:
        return None
    try:
        accelerators = [d for d in jax.devices() if d.platform.lower() in ("gpu", "tpu", "metal")]
        if not accelerators:
            return None
        first_accel = accelerators[0]
        accel_type = first_accel.platform.upper()
        return GpuProbeResult(accel_type, f"{accel_type} ({first_accel.device_kind}) via JAX")
    except Exception as e:
        logger.debug(f"JAX GPU probe failed unexpectedly: {e}")
    return None


# Fixed probe order preserves prior "first successful framework wins as primary" behavior.
GPU_PROBES: List[Tuple[str, Callable[[], Optional[GpuProbeResult]]]] = [
    ("torch", probe_torch_gpu),
    ("tensorflow", probe_tensorflow_gpu),
    ("jax", probe_jax_gpu),
]


def inspect_gpu_environment(imported_packages: Set[str]) -> Optional[GpuInfo]:
    """Coordinates per-framework GPU/accelerator probing across PyTorch, TensorFlow, and JAX."""
    expanded_imports = expand_transitive_frameworks(imported_packages)
    found_frameworks = list(SUPPORTED_GPU_FRAMEWORKS.intersection(expanded_imports))
    if not found_frameworks:
        return None

    framework_devices: Dict[str, Optional[str]] = {}
    active_types: List[str] = []
    primary_fw: Optional[str] = None
    primary_dev: Optional[str] = None

    for fw_stem, probe in GPU_PROBES:
        if fw_stem not in found_frameworks:
            continue
        result = probe()
        if result:
            framework_devices[fw_stem] = result.device_name
            active_types.append(result.accelerator_type)
            if not primary_dev:
                primary_fw = CANONICAL_TO_FRAMEWORK_DISPLAY.get(fw_stem, fw_stem.capitalize())
                primary_dev = result.device_name
        else:
            framework_devices[fw_stem] = None

    has_gpu = primary_dev is not None

    return GpuInfo(
        has_gpu=has_gpu,
        type=active_types[0] if active_types else None,
        active_framework=primary_fw,
        device_name=primary_dev,
        frameworks=sorted(found_frameworks),
        framework_devices=framework_devices
    )


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
    if gpu_info and gpu_info.has_gpu:
        dev_name = gpu_info.device_name
        active_fw = gpu_info.active_framework or "Framework"
        gpu_markdown_section = (
            f"- **Hardware Acceleration:** This notebook was created using a GPU accelerator (`{dev_name}`, verified via {active_fw}).\n"
            f"  If execution feels slow, ensure your runtime has a GPU accelerator enabled in environment settings."
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
                bullet_lines.append("    ⚠️ Specific hardware build tag detected. If installation fails, ensure your runtime matches this build.")
        local_builds_section = f"- **Specific Package Builds Detected:** The following package(s) use custom or hardware-specific builds:\n" + "\n".join(bullet_lines)

    markdown_lines = [
        "### 🛠️ Environment Setup & Dependency Verification",
        f"This notebook includes a pinned dependency list (`{DEFAULT_PINNED_MANIFEST_NAME}`) to ensure reproducible execution.\n",
        "- **Automatic Setup:** Cell 2 will verify your Python version and apply the exact package versions used by the author."
    ]
    
    if gpu_markdown_section:
        markdown_lines.append(gpu_markdown_section)
    if local_builds_section:
        markdown_lines.append(local_builds_section)
        
    markdown_lines.append("- **Network Notice:** Active internet access is required to download uncached packages.")

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
    print(f"If installation fails, consider changing your runtime Python version back to {{req_ver}}.\\n")

# Write explicit library requirements to a local file
requirements_content = \"\"\"# Tested top-level packages for this notebook
{payload_string}
\"\"\"

with open("{DEFAULT_PINNED_MANIFEST_NAME}", "w") as f:
    f.write(requirements_content.strip())

print(f"Applying pinned environment manifest [{timestamp}]...")
print("💡 Note: If installation pauses while offline, enable Internet access and re-run this cell.\\n")

# Run single-pass installation natively via pip
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-r", "{DEFAULT_PINNED_MANIFEST_NAME}"],
    capture_output=False
)

if result.returncode == 0:
    print("\\n✅ Primary dependencies installed successfully!")
    print("💡 If your notebook uses optional features or platform-specific tools, check the notes in pinned_requirements.txt.")
else:
    print("\\n❌ Setup failed while installing required packages.\\n")
    print("Troubleshooting Steps:")
    print("1. Internet Access: Ensure your notebook environment has active internet access.")
    print("2. GPU / Hardware: If using PyTorch or TensorFlow, check that GPU acceleration is enabled.")
    print("3. Detailed Error: Scroll up above this line to inspect detailed pip error logs.")
    print(f"\\n👉 For a detailed guide on resolving setup errors, see: {HELP_URL}")"""

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
    """Aggregates notebook scan results across a repository directory."""
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
    """Deterministically selects primary index URL based on repository frequency."""
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
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in DEFAULT_IGNORED_DIRS]
        for file in sorted(files):
            if file.endswith('.ipynb'):
                full_path = Path(root) / file
                ext_res = extract_from_file(str(full_path), strict=True)
                writefile_imports = extract_writefile_imports_from_sources(ext_res.code_sources)
                
                h_res = harvest_cell_magics_and_commands(ext_res.code_sources)
                harvested_urls = h_res.base_index_urls.union(h_res.extra_index_urls)

                parse_err = ext_res.error_msg if (not ext_res.success and "Skipped non-Python notebook" not in (ext_res.error_msg or "")) else None                
                res = NotebookScanResult(
                    path=full_path,
                    is_python=ext_res.success,
                    lang_label=ext_res.lang_label,
                    parse_error=parse_err,
                    imports=ext_res.imports,
                    submodules=ext_res.submodules,
                    guarded_imports=ext_res.guarded_imports,
                    dynamic_warnings=ext_res.dynamic_warnings,
                    code_sources=ext_res.code_sources,
                    harvested_urls=harvested_urls,
                    writefile_imports=writefile_imports,
                    harvested_pkgs=h_res.harvested_packages,
                    base_index_urls=h_res.base_index_urls,
                    extra_index_urls=h_res.extra_index_urls,
                    magic_warnings=h_res.magic_warnings,
                    magic_notices=h_res.magic_notices
                )
                repo_map.add_result(res)

    return repo_map


def analyze_batch_repository(
    repo_map: RepoEnvironmentMap, 
    frozen_env: Dict[str, str], 
    pkg_dist_map: Dict[str, List[str]], 
    batch_hw_cache: Optional[GpuInfo]
) -> BatchAnalysisSummary:
    """
    Aggregates dependency metrics, warnings, and index settings across repository notebooks.

    Args:
        repo_map: Populated RepoEnvironmentMap object from scanning target directory.
        frozen_env: Dict mapping package names to pinned freeze strings (pkg -> pkg==ver).
        pkg_dist_map: Dict mapping import stems to distribution names from importlib metadata.
        batch_hw_cache: Host GPU/accelerator inspection cache object.

    Returns:
        BatchAnalysisSummary containing aggregated counts, matched/missing packages, and warnings.
    """
    summary = BatchAnalysisSummary(
        target_dir=repo_map.target_dir,
        total_python_notebooks=len(repo_map.scan_results),
        non_python_count=len(repo_map.non_python_files),
        parse_errors=repo_map.parse_errors,
        batch_hw_cache=batch_hw_cache
    )

    for item in repo_map.non_python_files:
        summary.non_python_languages[item.lang_label] = (
            summary.non_python_languages.get(item.lang_label, 0) + 1
        )

    for res in repo_map.scan_results:
        nb_local_mods = get_notebook_local_modules(res.path, repo_map.target_dir)

        pinned_entries, notes = build_manifest_entries(
            res.imports, 
            res.submodules, 
            frozen_env, 
            pkg_dist_map, 
            guarded_imports=res.guarded_imports,
            local_repo_modules=nb_local_mods
        )

        _, _, hw_warns = process_package_requirements(
            pinned_entries, 
            res.harvested_urls, 
            base_urls=res.base_index_urls
        )
        for hw_pkg in hw_warns:
            summary.batch_hardware_warnings.setdefault(hw_pkg, []).append(res.path.name)

        for note in notes:
            if note not in summary.promotions:
                summary.promotions.append(note)

        for warn in res.dynamic_warnings:
            if warn not in summary.dynamic_warnings:
                summary.dynamic_warnings.append(warn)

        for warn in res.magic_warnings:
            if warn not in summary.magic_warnings:
                summary.magic_warnings.append(warn)

        for notice in res.magic_notices:
            if notice not in summary.magic_notices:
                summary.magic_notices.append(notice)

        for pin_entry in pinned_entries:
            if pin_entry.startswith("#"):
                if "provided automatically" in pin_entry or "local folder/file" in pin_entry:
                    continue
                pypi_name = pin_entry.split()[1]
                summary.missing_packages.setdefault(pypi_name, []).append(res.path.name)
            else:
                pkg_name = pin_entry.split("==")[0]
                summary.matched_packages.add(pkg_name)

        for pkg in res.harvested_pkgs:
            if pkg in STD_LIB or pkg in PLATFORM_PSEUDO_MODULES or pkg in nb_local_mods:
                continue
            pypi_name = IMPORT_TO_PYPI_MAP.get(pkg, pkg)
            matched_pin = frozen_env.get(pypi_name.lower())
            if matched_pin:
                pkg_name = matched_pin.split("==")[0]
                summary.matched_packages.add(pkg_name)
            else:
                summary.missing_packages.setdefault(pypi_name, []).append(res.path.name)

    primary_url, url_reason = select_primary_index_url(repo_map.url_to_notebooks)
    summary.primary_url = primary_url
    summary.primary_url_reason = url_reason

    return summary


def format_batch_report(summary: BatchAnalysisSummary) -> str:
    """Formats a BatchAnalysisSummary into a human-readable stdout report string."""
    out = []
    out.append("=" * 80)
    out.append("REPOSITORY REPRODUCIBILITY SUMMARY")
    out.append(f"Target Directory: {summary.target_dir}")
    out.append(f"Active Interpreter: {sys.executable}")
    out.append("=" * 80 + "\n")

    out.append("📁 NOTEBOOK INVENTORY & LANGUAGE SCAN:")
    out.append(f"  • Python (.ipynb): {summary.total_python_notebooks} files analyzed")
    
    if summary.non_python_count > 0:
        lang_str = ", ".join([f"{k} ({v})" for k, v in summary.non_python_languages.items()])
        out.append(f"  • Non-Python skipped: {summary.non_python_count} files [{lang_str}]")
    else:
        out.append("  • Non-Python skipped: 0 files")

    err_count = len(summary.parse_errors)
    out.append(f"  • File / Parse Errors: {err_count} files")
    out.append("")

    if err_count > 0:
        out.append("❌ FILE & PARSE ERRORS:")
        for err_res in summary.parse_errors:
            out.append(f"  • {err_res.path}")
            out.append(f"    └─ Cause: {err_res.parse_error}")
        out.append("")

    out.append(f"📦 REPOSITORY PACKAGE SUMMARY (Across {summary.total_python_notebooks} Python notebooks):")
    matched_list = sorted(summary.matched_packages)
    out.append(f"  • Installed & Verified: {len(matched_list)} packages ({', '.join(matched_list[:5])}{'...' if len(matched_list) > 5 else ''})")
    
    if summary.missing_packages:
        out.append(f"  • Packages missing from current environment: {len(summary.missing_packages)}")
        out.append("    (Action: Run 'pip install <package>' in active environment before generating lockfiles)")
        for pkg, nbs in sorted(summary.missing_packages.items()):
            nb_list = ", ".join(sorted(set(nbs))[:3])
            more = f", +{len(set(nbs))-3} more" if len(set(nbs)) > 3 else ""
            out.append(f"      - {pkg} (imported in: {nb_list}{more})")
    else:
        out.append("  • Packages missing from current environment: 0")
    out.append("")

    if summary.dynamic_warnings or summary.magic_warnings:
        out.append("⚠️ NOTICES & WARNINGS:")
        for warn in summary.dynamic_warnings:
            out.append(f"  • {warn}")
        for warn in summary.magic_warnings:
            out.append(f"  • {warn}")
        out.append("")

    if summary.magic_notices:
        out.append("ℹ️ SYSTEM & CONDA COMMANDS:")
        for notice in summary.magic_notices:
            out.append(f"  • {notice}")
        out.append("")

    if summary.promotions:
        out.append("💡 AUTOMATIC EXTRA PROMOTIONS:")
        for note in summary.promotions:
            out.append(f"  • {note}")
        out.append("")

    out.append("⚡ ACCELERATOR & DOWNLOAD INDEX CHECK:")
    if summary.batch_hw_cache and summary.batch_hw_cache.has_gpu:
        out.append(f"  • Active Hardware Accelerator: {summary.batch_hw_cache.device_name}")
    else:
        out.append("  • Active Hardware Accelerator: None (CPU-only execution environment)")

    if summary.primary_url:
        out.append(f"  • Primary Index URL: {summary.primary_url}")
        out.append(f"    └─ Selection Rule: {summary.primary_url_reason}")
    else:
        out.append("  • Extra Index URLs Harvested: None")

    if summary.batch_hardware_warnings:
        out.append("  • Custom Build Tag Warnings:")
        for pkg, nbs in sorted(summary.batch_hardware_warnings.items()):
            nb_list = ", ".join(sorted(set(nbs))[:3])
            more = f", +{len(set(nbs))-3} more" if len(set(nbs)) > 3 else ""
            out.append(f"      ⚠️ {pkg} (in: {nb_list}{more}) — No download URL harvested in code cells.")

    out.append("\n" + "-" * 80)
    if err_count > 0:
        out.append("STATUS: ⚠️ ATTENTION REQUIRED - Resolve file/parse errors above before building manifests.")
    else:
        out.append(f"STATUS: Ready. All {summary.total_python_notebooks} Python notebooks parsed successfully.")
    out.append("=" * 80)

    return "\n".join(out)


def generate_batch_analysis_report(
    repo_map: RepoEnvironmentMap, 
    frozen_env: Dict[str, str], 
    pkg_dist_map: Dict[str, List[str]], 
    batch_hw_cache: Optional[GpuInfo]
) -> Tuple[str, bool]:
    """Orchestrates batch repository analysis and returns (report_text, is_clean)."""
    summary = analyze_batch_repository(repo_map, frozen_env, pkg_dist_map, batch_hw_cache)
    report_text = format_batch_report(summary)
    return report_text, summary.is_clean


def generate_universal_manifest(
    repo_map: RepoEnvironmentMap, frozen_env: Dict[str, str], pkg_dist_map: Dict[str, List[str]]
) -> str:
    """Generates content string for universal manifest."""
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
        nb_local_mods = get_notebook_local_modules(res.path, repo_map.target_dir)
        entries, _ = build_manifest_entries(
            res.imports, 
            res.submodules, 
            frozen_env, 
            pkg_dist_map, 
            guarded_imports=res.guarded_imports,
            local_repo_modules=nb_local_mods
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
    local_repo_modules: Optional[Set[str]] = None,
    root_dir: Optional[str] = None
) -> Path:
    """
    Writes per-notebook locked file or replaces setup cells in-place.

    Args:
        scan_res: NotebookScanResult for the target notebook.
        frozen_env: Dict mapping package names to pinned freeze strings.
        pkg_dist_map: Dict mapping import stems to distribution names.
        batch_hw_cache: Host GPU/accelerator inspection cache object.
        suffix: Output file suffix if in_place is False (default: '_merged').
        in_place: If True, replaces setup cells in original notebook file.
        local_repo_modules: Explicit local module stems set to ignore.
        root_dir: Repository root directory string for scoped local module resolution.

    Returns:
        Path object pointing to written file.
    """
    if local_repo_modules is None:
        local_repo_modules = get_notebook_local_modules(scan_res.path, root_dir)

    pinned_manifest, _ = build_manifest_entries(
        scan_res.imports, 
        scan_res.submodules, 
        frozen_env, 
        pkg_dist_map, 
        guarded_imports=scan_res.guarded_imports,
        local_repo_modules=local_repo_modules
    )
    # scan_res.harvested_pkgs / base_index_urls / extra_index_urls were already
    # populated by harvest_cell_magics_and_commands() during the initial scan
    # (walk_and_scan_directory for batch mode, run_single_file_pipeline for
    # single-file mode) — reuse them rather than re-running the same
    # regex/tokenize pass over every code cell a second time.
    logger.debug(
        f"apply_output_to_notebook for '{scan_res.path.name}': reusing "
        f"{len(scan_res.harvested_pkgs)} pre-harvested package(s) and "
        f"{len(scan_res.base_index_urls) + len(scan_res.extra_index_urls)} index URL(s) "
        "from the initial scan (no re-harvest)."
    )
    aux_entries = build_auxiliary_tool_entries(scan_res.harvested_pkgs, scan_res.imports, frozen_env)
    writefile_entries = build_writefile_tool_entries(scan_res.writefile_imports, scan_res.imports, frozen_env)
    
    manifest_lines, local_tagged, _ = process_package_requirements(
        pinned_manifest, scan_res.harvested_urls, base_urls=scan_res.base_index_urls, auxiliary_entries=aux_entries, writefile_entries=writefile_entries
    )
    
    gpu_info: Optional[GpuInfo] = None
    if batch_hw_cache:
        expanded_nb_imports = expand_transitive_frameworks(scan_res.imports)
        nb_fw = set(batch_hw_cache.frameworks).intersection(expanded_nb_imports)
        if nb_fw:
            fw_devices = batch_hw_cache.framework_devices
            matched_fw = None
            matched_device = None

            for fw_stem in sorted(nb_fw):
                if fw_devices.get(fw_stem):
                    matched_fw = fw_stem
                    matched_device = fw_devices[fw_stem]
                    break

            if matched_device:
                active_label = CANONICAL_TO_FRAMEWORK_DISPLAY.get(matched_fw, matched_fw.capitalize())

                gpu_info = GpuInfo(
                    has_gpu=True,
                    type=batch_hw_cache.type,
                    active_framework=active_label,
                    device_name=matched_device,
                    frameworks=sorted(nb_fw),
                    framework_devices=fw_devices
                )
            else:
                gpu_info = GpuInfo(
                    has_gpu=False,
                    type=None,
                    active_framework=None,
                    device_name=None,
                    frameworks=sorted(nb_fw),
                    framework_devices=fw_devices
                )

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


def run_batch_pipeline(
    target_batch_dir: str, 
    args: argparse.Namespace, 
    frozen_env: Dict[str, str], 
    pkg_dist_map: Dict[str, List[str]], 
    batch_hw_cache: Optional[GpuInfo]
) -> None:
    """Executes the batch processing pipeline across a directory of notebooks."""
    repo_map = walk_and_scan_directory(target_batch_dir)
    report_text, is_clean = generate_batch_analysis_report(repo_map, frozen_env, pkg_dist_map, batch_hw_cache)
    print(report_text)

    if not is_clean and (args.universal or args.output or args.in_place):
        logger.error("\n❌ Execution aborted: Resolve file/parse errors before running --universal, --output, or --in-place.")
        sys.exit(1)

    if args.universal:
        manifest_filename = args.universal if isinstance(args.universal, str) else DEFAULT_UNIVERSAL_MANIFEST_NAME
        uni_content = generate_universal_manifest(repo_map, frozen_env, pkg_dist_map)
        out_file = Path(target_batch_dir) / manifest_filename
        with open(out_file, 'w', encoding='utf-8') as f:
            f.write(uni_content)
        logger.info(f"\n✅ Wrote universal repository manifest to '{out_file}'")

    if args.output or args.in_place:
        logger.info(f"\n🚀 Writing per-notebook locked files ({'in-place' if args.in_place else 'suffix: ' + args.suffix})...")
        for res in repo_map.scan_results:
            nb_local_mods = get_notebook_local_modules(res.path, repo_map.target_dir)
            written_path = apply_output_to_notebook(
                res, 
                frozen_env, 
                pkg_dist_map, 
                batch_hw_cache, 
                suffix=args.suffix, 
                in_place=args.in_place,
                local_repo_modules=nb_local_mods,
                root_dir=repo_map.target_dir
            )
            logger.info(f"  • Updated '{written_path.name}'")
        logger.info("✅ Batch output complete.")

    sys.exit(0)


def run_single_file_pipeline(
    args: argparse.Namespace, 
    frozen_env: Dict[str, str], 
    raw_full_freeze: List[str],
    pkg_dist_map: Dict[str, List[str]],
    precomputed_gpu_info: Optional[GpuInfo] = None
) -> None:
    """
    Executes single-notebook analysis or live IPython kernel history extraction.

    Args:
        precomputed_gpu_info: GPU inspection result already computed by main()
            from this same notebook file's imports (Path A only). Reused here
            to avoid probing GPU frameworks twice. Ignored for Path B (live
            kernel), where imports aren't known until session history is read,
            so GPU inspection is always run fresh in that branch.
    """
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
        
        ext_res = extract_from_file(args.notebook, strict=False)
        if not ext_res.success:
            logger.error(f"❌ Error: {ext_res.error_msg}")
            sys.exit(1)
            
        imports, submodules, code_sources = ext_res.imports, ext_res.submodules, ext_res.code_sources
        guarded_imports, dyn_warnings = ext_res.guarded_imports, ext_res.dynamic_warnings
        writefile_imports = extract_writefile_imports_from_sources(code_sources)
        gpu_info = precomputed_gpu_info
    elif in_live_ipython:
        logger.info("🔍 [Path B] Analyzing live IPython session kernel history via AST...")
        imports, submodules, code_sources, guarded_imports, dyn_warnings = extract_from_active_session()
        writefile_imports = extract_writefile_imports_from_sources(code_sources)
        gpu_info = inspect_gpu_environment(imports)
    else:
        return

    h_res = harvest_cell_magics_and_commands(code_sources)
    harvested_pkgs = h_res.harvested_packages
    base_urls, extra_urls = h_res.base_index_urls, h_res.extra_index_urls
    magic_warns, magic_notices = h_res.magic_warnings, h_res.magic_notices
    harvested_urls = extra_urls.union(base_urls)

    all_warnings = dyn_warnings + magic_warns

    pinned_manifest, promotion_notices = build_manifest_entries(
        imports, 
        submodules, 
        frozen_env, 
        pkg_dist_map, 
        guarded_imports=guarded_imports,
        local_repo_modules=single_file_local_modules
    )
    
    aux_entries = build_auxiliary_tool_entries(harvested_pkgs, imports, frozen_env)
    writefile_entries = build_writefile_tool_entries(writefile_imports, imports, frozen_env)

    manifest_lines, local_tagged_info, warnings = process_package_requirements(
        pinned_manifest, harvested_urls, base_urls=base_urls, auxiliary_entries=aux_entries, writefile_entries=writefile_entries
    )
    full_freeze_lines = raw_full_freeze if args.full_freeze else None

    if warnings:
        logger.warning("⚠️ HARDWARE BUILD WARNINGS:")
        for pkg in warnings:
            logger.warning(f"  • Specific hardware build detected: `{pkg}`")
            logger.warning("    No matching download URL was found in code cells. Ensure target machines match this build or supply an --extra-index-url.\n")

    if all_warnings:
        for warn in all_warnings:
            logger.warning(f"{warn}")
        logger.warning("")

    if magic_notices:
        for notice in magic_notices:
            logger.info(f"{notice}")
        logger.info("")

    if gpu_info:
        if gpu_info.has_gpu:
            logger.info(f"⚡ Active accelerator detected: {gpu_info.device_name}\n")
        elif gpu_info.frameworks:
            fw_list = ", ".join(gpu_info.frameworks)
            logger.warning(f"⚠️ Acceleration Framework ({fw_list}) imported, but NO active accelerator detected in host runtime.\n")

    if promotion_notices:
        for note in promotion_notices:
            logger.info(note)
        logger.info("")

    single_res = NotebookScanResult(
        path=Path(args.notebook) if args.notebook and not os.path.isdir(args.notebook) else Path("session.ipynb"),
        is_python=True,
        lang_label=StatusLabel.PYTHON,
        imports=imports,
        submodules=submodules,
        guarded_imports=guarded_imports,
        dynamic_warnings=dyn_warnings,
        code_sources=code_sources,
        harvested_urls=harvested_urls,
        writefile_imports=writefile_imports,
        harvested_pkgs=harvested_pkgs,
        base_index_urls=base_urls,
        extra_index_urls=extra_urls,
        magic_warnings=magic_warns,
        magic_notices=magic_notices
    )

    if args.output or args.in_place:
        logger.info(f"🚀 Writing updated notebook ({'in-place' if args.in_place else 'suffix: ' + args.suffix})...")
        written_path = apply_output_to_notebook(
            single_res,
            frozen_env,
            pkg_dist_map,
            gpu_info,
            suffix=args.suffix,
            in_place=args.in_place,
            local_repo_modules=single_file_local_modules,
            root_dir=target_single_file_dir
        )
        logger.info(f"✅ Updated '{written_path.name}'")
        sys.exit(0)

    blueprint = generate_production_blueprint(
        manifest_lines, 
        full_freeze_lines=full_freeze_lines, 
        local_tagged_info=local_tagged_info,
        gpu_info=gpu_info
    )

    print("--- [ STEP 1: PASTE INTO CELL 1 (MARKDOWN) ] ---\n")
    print(blueprint["step1_markdown"])
    print("\n" + "="*80 + "\n")

    print("--- [ STEP 2: PASTE INTO CELL 2 (CODE) ] ---\n")
    print(blueprint["step2_code"])
    print("\n" + "="*80)


def main() -> None:
    """CLI entrypoint and dispatch router for single notebook or batch analysis modes."""
    # This tool can run repeatedly inside a single long-lived process (a live
    # IPython kernel — see "Live IPython Kernel" in the module docstring),
    # where the notebook's own files may legitimately change between calls
    # to main(). Clear every memoized function's cache unconditionally at
    # the start of each call so caching (see _memoize_for_run) stays scoped
    # to a single logical run and never returns stale results across runs.
    get_notebook_local_modules.cache_clear()
    build_manifest_entries.cache_clear()

    parser = argparse.ArgumentParser(description="Generate environment lockfiles for Jupyter Notebooks.")
    parser.add_argument("notebook", nargs="?", help="Path to target .ipynb file or directory (when using --batch).")
    parser.add_argument("--full-freeze", action="store_true", help="Append full environment pip freeze after targeted manifest.")
    parser.add_argument("--quiet", action="store_true", help="Suppress diagnostic and status logging outputs.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose debug output.")
    
    # Batch / Output Flags
    parser.add_argument("--batch", metavar="DIR", help="Run in batch mode across all notebooks in specified directory.")
    parser.add_argument("--analyze", action="store_true", help="Run batch analysis mode (default when --batch is provided).")
    parser.add_argument(
        "--universal", 
        nargs="?", 
        const=DEFAULT_UNIVERSAL_MANIFEST_NAME, 
        default=None, 
        metavar="FILENAME",
        help=f"Generate universal repository manifest (default: '{DEFAULT_UNIVERSAL_MANIFEST_NAME}' when flag is provided)."
    )
    parser.add_argument("--output", action="store_true", help="Generate per-notebook merged lockfiles.")
    parser.add_argument("--suffix", default="_merged", help="File suffix for merged notebook outputs (default: '_merged').")
    parser.add_argument("--in-place", action="store_true", help="Overwrite original notebooks in-place instead of creating companion files.")

    args, unknown = parser.parse_known_args()

    if args.quiet:
        logger.setLevel(logging.ERROR)
    elif args.verbose:
        logger.setLevel(logging.DEBUG)

    if (args.output or args.in_place) and not args.batch and not args.notebook:
        logger.error("❌ Error: --output or --in-place requires a target notebook file path or --batch directory.")
        sys.exit(1)

    frozen_env, raw_full_freeze = get_installed_environment()
    pkg_dist_map = importlib.metadata.packages_distributions() if hasattr(importlib.metadata, "packages_distributions") else {}
    
    target_batch_dir = args.batch or (args.notebook if args.notebook and os.path.isdir(args.notebook) else None)
    initial_imports: Set[str] = set()

    if target_batch_dir:
        repo_map_pre = walk_and_scan_directory(target_batch_dir)
        initial_imports.update(repo_map_pre.global_imports)
    elif args.notebook and os.path.isfile(args.notebook):
        ext_res = extract_from_file(args.notebook, strict=False)
        initial_imports.update(ext_res.imports)

    # Only probe GPU frameworks that are actually imported somewhere in the target
    # notebook(s) — inspect_gpu_environment returns None early if initial_imports
    # has no overlap with SUPPORTED_GPU_FRAMEWORKS, avoiding an unconditional
    # `import torch`/`tensorflow`/`jax` on every run regardless of notebook content.
    batch_hw_cache = inspect_gpu_environment(initial_imports)

    if target_batch_dir:
        run_batch_pipeline(target_batch_dir, args, frozen_env, pkg_dist_map, batch_hw_cache)
    else:
        # For Path A (saved notebook file), batch_hw_cache above was already computed
        # from this same file's imports, so it's reused rather than re-probed.
        # For Path B (live IPython kernel), imports aren't known until
        # run_single_file_pipeline extracts kernel history, so it probes fresh there.
        run_single_file_pipeline(args, frozen_env, raw_full_freeze, pkg_dist_map, batch_hw_cache)


if __name__ == "__main__":
    main()
    if target_batch_dir:
        repo_map_pre = walk_and_scan_directory(target_batch_dir)
        initial_imports.update(repo_map_pre.global_imports)
    elif args.notebook and os.path.isfile(args.notebook):
        ext_res = extract_from_file(args.notebook, strict=False)
        initial_imports.update(ext_res.imports)

    # Only probe GPU frameworks that are actually imported somewhere in the target
    # notebook(s) — inspect_gpu_environment returns None early if initial_imports
    # has no overlap with SUPPORTED_GPU_FRAMEWORKS, avoiding an unconditional
    # `import torch`/`tensorflow`/`jax` on every run regardless of notebook content.
    batch_hw_cache = inspect_gpu_environment(initial_imports)

    if target_batch_dir:
        run_batch_pipeline(target_batch_dir, args, frozen_env, pkg_dist_map, batch_hw_cache)
    else:
        # For Path A (saved notebook file), batch_hw_cache above was already computed
        # from this same file's imports, so it's reused rather than re-probed.
        # For Path B (live IPython kernel), imports aren't known until
        # run_single_file_pipeline extracts kernel history, so it probes fresh there.
        run_single_file_pipeline(args, frozen_env, raw_full_freeze, pkg_dist_map, batch_hw_cache)


if __name__ == "__main__":
    main()
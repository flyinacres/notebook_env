#!/usr/bin/env python3
"""
notebook_env.py (v43)
Headless Jupyter Notebook Dependency Scanner & Lockfile Generator.

Standalone, zero-dependency utility for analyzing notebook environments,
detecting GPU/accelerator requirements, harvesting scoped index URLs, and emitting
reproducible lockfile manifests and isolated sequential installation blueprints.

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
  1. Single Notebook CLI:  python notebook_env.py notebook.ipynb [--format {text,json}] [--output | --output-dir DIR | --in-place]
  2. Batch Repo Directory: python notebook_env.py --batch ./repo [--format {text,json}] [--universal [FILENAME]] [--output | --output-dir DIR | --in-place]
  3. Live IPython Kernel:   import notebook_env as ne; ne.main()
"""

# =====================================================================
# CONSTANTS, LOGGING & TYPE DEFINITIONS
# =====================================================================

import ast
import json
import os
import re
import sys
import argparse
import contextlib
import functools
import logging
import warnings
import subprocess
import importlib.metadata
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Set, Dict, List, Tuple, Optional, Any, TypedDict, Callable, NamedTuple, Union

TOOL_VERSION: str = "43"
SCHEMA_VERSION: str = "1.0"

# Force UTF-8 encoding for stdout and stderr on Windows/redirected environments
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Setup logging stream for diagnostic messages (directed to stderr)
logger = logging.getLogger("notebook_env")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)


# =====================================================================
# SYSTEM CONSTANTS & CONFIGURATION DEFAULTS
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
class DiagnosticEvent:
    """Represents a structured warning or informational notice."""
    type: str
    detail: str
    cell_idx: Optional[int] = None
    line_idx: Optional[int] = None
    level: str = "warning"

    def format_console(self) -> str:
        prefix = "⚠️" if self.level == "warning" else "ℹ️"
        return f"{prefix} {self.detail}"

    def __str__(self) -> str:
        return self.format_console()

    def __contains__(self, item: str) -> bool:
        return item in self.detail or item in self.format_console()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "detail": self.detail,
            "cell_idx": self.cell_idx,
            "line_idx": self.line_idx,
        }


@dataclass
class PromotionDetail:
    """Represents an automatic extras package promotion event."""
    import_name: str
    promoted_name: str
    version: Optional[str] = None
    detail: str = ""

    def __str__(self) -> str:
        return self.detail

    def __contains__(self, item: str) -> bool:
        return item in self.detail or item in self.promoted_name or item in self.import_name

    def to_dict(self) -> Dict[str, Any]:
        return {
            "import": self.import_name,
            "promoted_name": self.promoted_name,
            "version": self.version,
        }


@dataclass
class PipInstallOccurrence:
    """
    Represents an explicit package install invocation inside a notebook cell.
    
    Fields:
        cell_idx: Index of the cell (document or execution order rank).
        line_idx: Line index inside the cell where the install was declared.
        raw_token: Raw CLI argument token (e.g. 'torch==2.3.1+cu121').
        name: PyPI distribution stem (e.g. 'torch').
        version_spec: Version specifier string if present (e.g. '==2.3.1+cu121', '>=2.0'), else empty.
        flags: Scoped CLI flags accompanying this specific command.
    """
    cell_idx: int
    line_idx: int
    raw_token: str
    name: str
    version_spec: str = ""
    flags: List[str] = field(default_factory=list)


@dataclass
class ImportOccurrence:
    """
    Represents an AST import statement inside a notebook cell.
    
    Fields:
        cell_idx: Index of the cell.
        line_idx: Zero-indexed line number in the original cell source.
        module: Base imported package stem (e.g. 'torch').
        full_name: Full imported module string (e.g. 'umap.plot').
        is_guarded: True if import occurred inside a try/except or conditional block.
    """
    cell_idx: int
    line_idx: int
    module: str
    full_name: str = ""
    is_guarded: bool = False


@dataclass
class DependencyEntry:
    """
    Represents a single dependency requirement with scoped flags or an informational comment.

    Fields:
        name: Distribution / PyPI package name (e.g., 'torch', 'pandas').
        version: Pinned version string (e.g., '2.3.1+cu121', '2.2.1') or empty if unversioned.
        flags: Scoped CLI flags to pass to pip install (e.g., ['--extra-index-url', 'https://...']).
        source: Discovery origin ('import', 'pip_command', 'writefile_script').
        status: Semantic category ('pinned', 'guarded', 'platform_pseudo_module', 'build_tool', 'local_module', 'auxiliary_tool', 'writefile_script').
        is_comment: True if this entry represents a comment, platform pseudo-module, or uninstalled fallback.
        comment_text: Full string representation when is_comment is True.
    """
    name: str = ""
    version: str = ""
    flags: List[str] = field(default_factory=list)
    source: str = "import"
    status: str = "pinned"
    is_comment: bool = False
    comment_text: str = ""

    @property
    def specifier(self) -> str:
        """Returns the pip install specifier (e.g. 'pandas==2.2.1')."""
        if self.is_comment:
            return self.comment_text
        return f"{self.name}=={self.version}" if self.version else self.name

    def to_dict(self) -> Dict[str, Any]:
        """Converts to a dictionary representation for Cell 2 inline execution metadata."""
        return {
            "name": self.name,
            "version": self.version,
            "flags": self.flags
        }

    def to_report_dict(self) -> Dict[str, Any]:
        """Converts to full JSON report representation."""
        return {
            "name": self.name,
            "version": self.version if self.version else None,
            "source": self.source,
            "status": self.status,
            "hardware_tagged": ("+" in self.version) if self.version else False,
            "flags": self.flags,
            "comment": self.comment_text if self.is_comment else None
        }


@dataclass
class TimelineResult:
    """
    Encapsulates the resolved dependency timeline and associated promotion and conflict notices.

    Fields:
        dependencies: Ordered list of DependencyEntry objects.
        promotion_notices: Informational PromotionDetail objects.
        conflict_warnings: Structured diagnostic events for overwritten pins or conflicting flags.
    """
    dependencies: List[DependencyEntry] = field(default_factory=list)
    promotion_notices: List[PromotionDetail] = field(default_factory=list)
    conflict_warnings: List[DiagnosticEvent] = field(default_factory=list)

    def __iter__(self):
        """Unpacking fallback allowing `deps, notices = timeline_result` in legacy callers."""
        notice_strings = [p.detail for p in self.promotion_notices]
        return iter((self.dependencies, notice_strings))


@dataclass
class GpuInfo:
    """Payload representing active host accelerator capabilities across PyTorch, TensorFlow, and JAX."""
    has_gpu: bool = False
    type: Optional[str] = None
    active_framework: Optional[str] = None
    device_name: Optional[str] = None
    frameworks: List[str] = field(default_factory=list)
    framework_devices: Dict[str, Optional[str]] = field(default_factory=dict)
    probe_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_gpu": self.has_gpu,
            "framework": self.active_framework,
            "device_name": self.device_name,
            "frameworks_detected": self.frameworks,
            "probe_errors": self.probe_errors
        }


class BlueprintResult(TypedDict):
    """Cell blueprint output strings for Cell 1 (Markdown) and Cell 2 (Python script)."""
    step1_markdown: str
    step2_code: str


@dataclass
class NotebookAnalysisReport:
    """Standardized single-notebook analysis report object."""
    notebook_path: str
    is_python: bool
    lang_label: str
    parse_error: Optional[str] = None
    dependencies: List[DependencyEntry] = field(default_factory=list)
    local_modules: List[str] = field(default_factory=list)
    platform_pseudo_modules: List[str] = field(default_factory=list)
    build_and_packaging_tools: List[str] = field(default_factory=list)
    gpu: Optional[GpuInfo] = None
    warnings: List[DiagnosticEvent] = field(default_factory=list)
    notices: List[DiagnosticEvent] = field(default_factory=list)
    promotions: List[PromotionDetail] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "notebook_path": self.notebook_path,
            "is_python": self.is_python,
            "lang_label": self.lang_label,
            "parse_error": self.parse_error,
            "dependencies": [d.to_report_dict() for d in self.dependencies],
            "local_modules": self.local_modules,
            "platform_pseudo_modules": self.platform_pseudo_modules,
            "build_and_packaging_tools": self.build_and_packaging_tools,
            "gpu": self.gpu.to_dict() if self.gpu else None,
            "warnings": [w.to_dict() for w in self.warnings],
            "notices": [n.to_dict() for n in self.notices],
            "promotions": [p.to_dict() for p in self.promotions],
        }


@dataclass
class NotebookScanResult:
    """Complete AST and metadata analysis payload for an individual notebook file."""
    path: Path
    is_python: bool
    lang_label: str
    parse_error: Optional[str] = None
    imports: List[str] = field(default_factory=list)
    submodules: Dict[str, Set[str]] = field(default_factory=dict)
    guarded_imports: Set[str] = field(default_factory=set)
    dynamic_warnings: List[DiagnosticEvent] = field(default_factory=list)
    code_sources: List[str] = field(default_factory=list)
    harvested_urls: Optional[Set[str]] = None
    writefile_imports: List[str] = field(default_factory=list)
    harvested_pkgs: Set[str] = field(default_factory=set)
    base_index_urls: Set[str] = field(default_factory=set)
    extra_index_urls: Set[str] = field(default_factory=set)
    scoped_flags: Dict[str, List[str]] = field(default_factory=dict)
    magic_warnings: List[DiagnosticEvent] = field(default_factory=list)
    magic_notices: List[DiagnosticEvent] = field(default_factory=list)

    def __post_init__(self):
        if self.harvested_urls is None:
            if self.code_sources:
                self.harvested_urls = harvest_index_urls_from_sources(self.code_sources)
            else:
                self.harvested_urls = set()


@dataclass
class ExtractionResult:
    """Encapsulates the raw extraction payload from reading a notebook file."""
    success: bool
    lang_label: str
    imports: List[str] = field(default_factory=list)
    submodules: Dict[str, Set[str]] = field(default_factory=dict)
    code_sources: List[str] = field(default_factory=list)
    error_msg: Optional[str] = None
    guarded_imports: Set[str] = field(default_factory=set)
    dynamic_warnings: List[DiagnosticEvent] = field(default_factory=list)
    writefile_imports: List[str] = field(default_factory=list)

    def __iter__(self):
        """Legacy tuple-unpacking fallback for backward compatibility."""
        dyn_warn_strings = [w.format_console() for w in self.dynamic_warnings]
        return iter((
            self.success,
            self.imports,
            self.submodules,
            self.code_sources,
            self.error_msg,
            self.lang_label,
            self.guarded_imports,
            dyn_warn_strings,
        ))


@dataclass
class HarvestResult:
    """Encapsulates harvested packages, index URLs, scoped flags, warnings, and notices from cell magics."""
    harvested_packages: Set[str] = field(default_factory=set)
    base_index_urls: Set[str] = field(default_factory=set)
    extra_index_urls: Set[str] = field(default_factory=set)
    magic_warnings: List[DiagnosticEvent] = field(default_factory=list)
    magic_notices: List[DiagnosticEvent] = field(default_factory=list)
    scoped_flags: Dict[str, List[str]] = field(default_factory=dict)

    def __iter__(self):
        """Legacy tuple-unpacking fallback for backward compatibility."""
        warn_strings = [w.format_console() for w in self.magic_warnings]
        notice_strings = [n.format_console() for n in self.magic_notices]
        return iter((
            self.harvested_packages,
            self.base_index_urls,
            self.extra_index_urls,
            warn_strings,
            notice_strings,
        ))


@dataclass
class BatchAnalysisSummary:
    """Aggregated analysis metrics across all notebooks in a batch repo scan."""
    target_dir: str
    total_python_notebooks: int = 0
    non_python_count: int = 0
    non_python_languages: Dict[str, int] = field(default_factory=dict)
    companion_skipped_count: int = 0
    parse_errors: List[Dict[str, str]] = field(default_factory=list)
    matched_packages: Set[str] = field(default_factory=set)
    missing_packages: Dict[str, List[str]] = field(default_factory=dict)
    promotions: List[PromotionDetail] = field(default_factory=list)
    dynamic_warnings: List[DiagnosticEvent] = field(default_factory=list)
    magic_warnings: List[DiagnosticEvent] = field(default_factory=list)
    magic_notices: List[DiagnosticEvent] = field(default_factory=list)
    batch_hardware_warnings: Dict[str, List[str]] = field(default_factory=dict)
    primary_url: Optional[str] = None
    primary_url_reason: Optional[str] = None
    batch_hw_cache: Optional[GpuInfo] = None
    notebooks: List[NotebookAnalysisReport] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return len(self.parse_errors) == 0


# =====================================================================
# MAPPINGS, PLATFORM INJECTIONS & STDLIB LOOKUP
# =====================================================================

IMPORT_TO_PYPI_MAP: Dict[str, str] = {
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "PIL": "Pillow",
    "yaml": "PyYAML",
    "bs4": "beautifulsoup4",
    "attr": "attrs",
    "serial": "pyserial",
    "dotenv": "python-dotenv",
    "mpl_toolkits": "matplotlib",
    "skimage": "scikit-image"
}

# Standard build and packaging bootstrap tools; excluded from requirement lockfiles
BUILD_AND_PACKAGING_TOOLS: Set[str] = {
    "pip",
    "setuptools",
    "wheel"
}

PLATFORM_PSEUDO_MODULES: Set[str] = {
    "dbutils",
    "kaggle_secrets",
    "google.colab",
    "pyspark.dbutils",
    "__main__",
    "notebook_env",
    "databricks"
}

TRANSITIVE_FRAMEWORK_MAP: Dict[str, str] = {
    "fastai": "torch",
    "torchvision": "torch",
    "torchaudio": "torch",
    "timm": "torch",
    "keras": "tensorflow",
    "flax": "jax",
}

FRAMEWORK_NAME_TO_CANONICAL: Dict[str, str] = {
    v: k for k, v in CANONICAL_TO_FRAMEWORK_DISPLAY.items()
}

STD_LIB: Set[str] = set(sys.stdlib_module_names) if hasattr(sys, 'stdlib_module_names') else {
    "os", "sys", "re", "json", "ast", "subprocess", "datetime", "math", "random", 
    "time", "pathlib", "typing", "collections", "itertools", "functools", "shutil"
}


def canonicalize_pkg_name(name: str) -> str:
    """PEP 503 normalization: lowercase and replace runs of [-_.] with a single hyphen."""
    return re.sub(r"[-_.]+", "-", name).strip("-").lower()


def is_running_in_ipython() -> bool:
    """Checks whether execution is occurring inside an active IPython/Jupyter kernel."""
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


def sanitize_kernel_argv(args: argparse.Namespace) -> None:
    """
    Cleans up contaminated sys.argv from ipykernel launcher (e.g. ['-f', 'kernel-xxx.json']).
    Prevents Path A from attempting to parse connection JSON files.
    """
    if not args.notebook:
        return

    nb_str = str(args.notebook)
    if "kernel-" in nb_str and nb_str.endswith(".json"):
        args.notebook = None
    elif not nb_str.endswith(".ipynb") and is_running_in_ipython():
        if not os.path.exists(nb_str) or not (os.path.isdir(nb_str) or nb_str.endswith(".ipynb")):
            args.notebook = None


@contextlib.contextmanager
def silence_fd2_stderr():
    """
    Temporarily redirects OS-level file descriptor 2 (stderr) to os.devnull.
    Prevents low-level C++ drivers (e.g. CUDA cuInit 303) from polluting output.
    Ensures safe, generator-compliant exception propagation without crashing.
    """
    old_stderr_fd = None
    devnull_fd = None
    try:
        try:
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            old_stderr_fd = os.dup(2)
            os.dup2(devnull_fd, 2)
        except Exception:
            pass

        yield

    finally:
        if old_stderr_fd is not None:
            try:
                os.dup2(old_stderr_fd, 2)
                os.close(old_stderr_fd)
            except Exception:
                pass
        if devnull_fd is not None:
            try:
                os.close(devnull_fd)
            except Exception:
                pass


def get_timeline_context_label(is_execution_ordered: bool) -> str:
    """Returns the standardized authority qualifier for timeline-dependent diagnostics."""
    if is_execution_ordered:
        return "in execution sequence"
    return "in document order (execution counts unavailable or inconsistent)"


def get_ordered_code_cells(cells: List[Dict[str, Any]]) -> Tuple[List[Tuple[int, Dict[str, Any]]], bool]:
    """
    Evaluates execution_count across all code cells.
    If 100% of code cells have valid, unique positive integer execution counts,
    orders cells strictly by execution_count ascending.
    Otherwise, falls back 100% to document index order.
    """
    code_cells = [(idx, c) for idx, c in enumerate(cells) if c.get("cell_type") == "code"]
    if not code_cells:
        return [], False

    counts = [c.get("execution_count") for _, c in code_cells]
    
    is_fully_ordered = (
        all(isinstance(cnt, int) and cnt > 0 for cnt in counts)
        and len(set(counts)) == len(counts)
    )

    if is_fully_ordered:
        ordered = sorted(code_cells, key=lambda pair: pair[1]["execution_count"])
        return ordered, True

    return code_cells, False


def discover_local_repo_modules(target_dir: str) -> Set[str]:
    """Scans target_dir for valid top-level Python modules and packages to prevent false-positive PyPI warnings."""
    local_mods: Set[str] = set()
    target_path = Path(target_dir)
    if not target_path.exists():
        return local_mods

    try:
        for entry in target_path.iterdir():
            if entry.is_file() and entry.suffix == ".py" and entry.stem != "__init__":
                local_mods.add(entry.stem)
            elif entry.is_dir() and entry.name not in DEFAULT_IGNORED_DIRS and not entry.name.startswith('.'):
                if any(entry.rglob("*.py")):
                    local_mods.add(entry.name)
    except Exception as e:
        logger.debug(f"[AST/ModuleScan] Failed scanning '{target_dir}' for local modules: {e}")

    return local_mods


def _memoize_for_run(func: Callable) -> Callable:
    """Memoizes functions scoped to a single run, handling Set, List, and Dict arguments."""
    cache: Dict[Tuple[Any, ...], Any] = {}

    def _cache_key_part(value: Any) -> Any:
        if isinstance(value, dict):
            return id(value)
        if isinstance(value, set):
            return frozenset(value)
        if isinstance(value, list):
            return tuple(_cache_key_part(item) for item in value)
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
# =====================================================================

def detect_notebook_language(nb_data: Dict[str, Any], strict: bool = False) -> Tuple[bool, str]:
    """Inspects kernelspec and language_info metadata."""
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
    """Reads a Jupyter Notebook JSON file and extracts code sources, imports, guarded state, and dynamic warnings."""
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
    ordered_cells, _ = get_ordered_code_cells(cells)
    code_sources = ["".join(c.get("source", [])) for _, c in ordered_cells]
    imports, submodules, guarded_imports, dyn_warnings, writefile_imports = extract_imports_from_sources_full(code_sources)

    return ExtractionResult(
        success=True,
        lang_label=lang_label,
        imports=imports,
        submodules=submodules,
        code_sources=code_sources,
        guarded_imports=guarded_imports,
        dynamic_warnings=dyn_warnings,
        writefile_imports=writefile_imports
    )


def extract_from_active_session() -> Tuple[List[str], Dict[str, Set[str]], List[str], Set[str], List[DiagnosticEvent]]:
    """
    Path B (Live Kernel): Reads IPython execution history in chronological order.
    Filters out self-referential notebook_env execution cells and invocation commands.
    """
    import __main__
    raw_sources = [src for src in getattr(__main__, 'In', []) if src and isinstance(src, str)]
    
    clean_sources: List[str] = []
    for src in raw_sources:
        if "NotebookImportVisitor" in src or "def extract_from_active_session" in src:
            continue
        stripped = src.strip()
        if re.search(r'\b(?:ne|notebook_env)\.main\s*\(', stripped) or stripped == "import notebook_env" or stripped.startswith("import notebook_env as"):
            continue
        clean_sources.append(src)

    imports, submodules, guarded_imports, dyn_warnings = extract_imports_from_sources_typed(clean_sources)
    return imports, submodules, clean_sources, guarded_imports, dyn_warnings


# =====================================================================
# AST VISITOR & DYNAMIC IMPORT PARSER
# =====================================================================

class NotebookImportVisitor(ast.NodeVisitor):
    """AST visitor traversing Python code to record imports, guarded states, and dynamic calls in order."""
    def __init__(self, cell_idx: int = 0) -> None:
        self.cell_idx: int = cell_idx
        self.imports: List[str] = []
        self.writefile_imports: List[str] = []
        self.submodules: Dict[str, Set[str]] = {}
        self.unconditional_imports: Set[str] = set()
        self.raw_guarded_imports: Set[str] = set()
        self.dynamic_import_warnings: List[DiagnosticEvent] = []
        self.occurrences: List[ImportOccurrence] = []
        self._guarded_depth: int = 0
        self._in_writefile: bool = False

        self._importlib_aliases: Set[str] = {"importlib"}
        self._import_module_bindings: Set[str] = set()

    @property
    def guarded_imports(self) -> Set[str]:
        return self.raw_guarded_imports - self.unconditional_imports

    def _record_import(self, base_pkg: str, full_name: Optional[str] = None, lineno: int = 1) -> None:
        line_idx = max(0, lineno - 1)
        if self._in_writefile:
            if base_pkg not in self.writefile_imports:
                self.writefile_imports.append(base_pkg)
            return

        if base_pkg not in self.imports:
            self.imports.append(base_pkg)

        is_guarded = self._guarded_depth > 0
        if is_guarded:
            self.raw_guarded_imports.add(base_pkg)
        else:
            self.unconditional_imports.add(base_pkg)

        if full_name and '.' in full_name:
            self.submodules.setdefault(base_pkg, set()).add(full_name)

        self.occurrences.append(
            ImportOccurrence(
                cell_idx=self.cell_idx,
                line_idx=line_idx,
                module=base_pkg,
                full_name=full_name or base_pkg,
                is_guarded=is_guarded
            )
        )

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
            self._record_import(base_pkg, full_name=alias.name, lineno=node.lineno)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            base_pkg = node.module.split('.')[0]
            if node.module == "importlib":
                for alias in node.names:
                    if alias.name == "import_module":
                        self._import_module_bindings.add(alias.asname or "import_module")
            self._record_import(base_pkg, full_name=node.module, lineno=node.lineno)
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
                self._record_import(base_pkg, full_name=imported_pkg, lineno=node.lineno)
            else:
                expr_repr = ast.unparse(first_arg) if hasattr(ast, "unparse") else "expression"
                self.dynamic_import_warnings.append(
                    DiagnosticEvent(
                        type="dynamic_import",
                        detail=f"Dynamic import detected via variable '{expr_repr}'. Check that this package is installed if execution fails.",
                        cell_idx=self.cell_idx,
                        line_idx=getattr(node, "lineno", 1) - 1,
                        level="warning"
                    )
                )

        self.generic_visit(node)


def extract_import_occurrences_from_source(source: str, cell_idx: int = 0) -> List[ImportOccurrence]:
    """
    Parses an individual cell source using blank-line padding for stripped magics
    so that AST lineno perfectly matches raw cell line numbers.
    """
    cell_type, clean_body = classify_cell_source(source)
    if cell_type in {"SHELL_SCRIPT", "WRITEFILE"}:
        return []

    clean_lines = [
        "" if (line.strip().startswith('%') or line.strip().startswith('!')) else line
        for line in clean_body.splitlines()
    ]
    clean_source = "\n".join(clean_lines)

    visitor = NotebookImportVisitor(cell_idx=cell_idx)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=SyntaxWarning)
            tree = ast.parse(clean_source)
        visitor.visit(tree)
    except SyntaxError:
        return []

    return visitor.occurrences


def extract_imports_from_sources_full(
    code_sources: List[str]
) -> Tuple[List[str], Dict[str, Set[str]], Set[str], List[DiagnosticEvent], List[str]]:
    """Executes single-pass AST traversal returning primary and writefile imports with typed diagnostics."""
    visitor = NotebookImportVisitor()
    for cell_idx, source in enumerate(code_sources):
        visitor.cell_idx = cell_idx
        cell_type, clean_body = classify_cell_source(source)

        if cell_type == "SHELL_SCRIPT":
            continue

        visitor._in_writefile = (cell_type == "WRITEFILE")

        clean_lines = [
            "" if (line.strip().startswith('%') or line.strip().startswith('!')) else line
            for line in clean_body.splitlines()
        ]
        clean_source = "\n".join(clean_lines)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=SyntaxWarning)
                tree = ast.parse(clean_source)
            visitor.visit(tree)
        except SyntaxError:
            continue

    primary_imports = [imp for imp in visitor.imports if imp not in visitor.writefile_imports]
    return (
        primary_imports, 
        visitor.submodules, 
        visitor.guarded_imports, 
        visitor.dynamic_import_warnings,
        visitor.writefile_imports
    )


def extract_imports_from_sources(
    code_sources: List[str]
) -> Tuple[List[str], Dict[str, Set[str]], Set[str], List[str]]:
    """Legacy 4-tuple extractor for primary imports with formatted strings."""
    primary_imports, submodules, guarded, dyn_warns, _ = extract_imports_from_sources_full(code_sources)
    return primary_imports, submodules, guarded, [w.format_console() for w in dyn_warns]


def extract_imports_from_sources_typed(
    code_sources: List[str]
) -> Tuple[List[str], Dict[str, Set[str]], Set[str], List[DiagnosticEvent]]:
    """Typed 4-tuple extractor for primary imports returning DiagnosticEvent objects."""
    primary_imports, submodules, guarded, dyn_warns, _ = extract_imports_from_sources_full(code_sources)
    return primary_imports, submodules, guarded, dyn_warns


def extract_writefile_imports_from_sources(code_sources: List[str]) -> List[str]:
    """Extracts writefile script imports."""
    _, _, _, _, writefile_imports = extract_imports_from_sources_full(code_sources)
    return writefile_imports


# =====================================================================
# CELL MAGIC & OCCURRENCE HARVESTER
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

def harvest_pip_install_occurrences(code_sources: List[str]) -> List[PipInstallOccurrence]:
    """
    Walks all cell lines and extracts structured PipInstallOccurrence records.
    Filters out %%writefile cells completely.
    """
    occurrences: List[PipInstallOccurrence] = []

    for cell_idx, source in enumerate(code_sources):
        cell_type, clean_body = classify_cell_source(source)
        if cell_type == "WRITEFILE":
            continue

        for line_idx, line in enumerate(clean_body.splitlines()):
            clean_line = line.strip()
            if not clean_line or clean_line.startswith('#') or clean_line in SHELL_CELL_MAGICS:
                continue

            command_segments = SHELL_SPLIT_PATTERN.split(clean_line)
            for segment in command_segments:
                seg = segment.strip()
                pip_match = PIP_INSTALL_PATTERN.match(seg)
                if not pip_match:
                    continue

                args_str = pip_match.group(1)
                tokens = args_str.split()
                line_flags: List[str] = []
                token_specs: List[Tuple[str, str, str]] = []

                # Pass 1: Harvest all flags across the command segment first
                i = 0
                while i < len(tokens):
                    token = tokens[i]
                    if token in {"--extra-index-url", "--index-url", "-i", "-f", "--find-links"}:
                        if i + 1 < len(tokens):
                            line_flags.extend([token, tokens[i+1].strip("'\"")])
                            i += 2
                            continue
                    elif token in PIP_VALUE_FLAGS:
                        i += 2
                        continue
                    i += 1

                # Pass 2: Extract package names and specs
                i = 0
                while i < len(tokens):
                    token = tokens[i]
                    if token in {"--extra-index-url", "--index-url", "-i", "-f", "--find-links"} or token in PIP_VALUE_FLAGS:
                        i += 2
                        continue
                    elif token.startswith('-') or token.lower() in PIP_SINGLE_FLAGS:
                        i += 1
                        continue
                    elif any(token.lower().startswith(p) for p in VCS_OR_PATH_PREFIXES):
                        i += 1
                        continue

                    match = re.search(r'[<>=!~;\[#]', token)
                    if match:
                        split_idx = match.start()
                        pkg_name = token[:split_idx].strip("'\"")
                        v_spec = token[split_idx:].strip("'\"")
                    else:
                        pkg_name = token.strip("'\"")
                        v_spec = ""

                    if pkg_name:
                        token_specs.append((token, pkg_name, v_spec))
                    i += 1

                for raw_tok, pkg, v_spec in token_specs:
                    occurrences.append(
                        PipInstallOccurrence(
                            cell_idx=cell_idx,
                            line_idx=line_idx,
                            raw_token=raw_tok,
                            name=pkg,
                            version_spec=v_spec,
                            flags=list(line_flags)
                        )
                    )

    return occurrences


def resolve_pip_occurrences(
    occurrences: List[PipInstallOccurrence],
    is_execution_ordered: bool = True
) -> Tuple[Dict[str, PipInstallOccurrence], List[DiagnosticEvent]]:
    """
    Applies atomic last-wins resolution across occurrences.
    The later occurrence completely replaces earlier occurrences (name, version, flags indivisibly).
    Emits synchronized confidence-hedged warnings on pin or flag conflicts.
    """
    resolved: Dict[str, PipInstallOccurrence] = {}
    conflict_warnings: List[DiagnosticEvent] = []
    seen_history: Dict[str, List[PipInstallOccurrence]] = {}

    for occ in occurrences:
        norm_key = canonicalize_pkg_name(occ.name)
        seen_history.setdefault(norm_key, []).append(occ)

    time_qualifier = get_timeline_context_label(is_execution_ordered)

    for norm_key, history in seen_history.items():
        winning_occ = history[-1]
        resolved[norm_key] = winning_occ
        resolved[winning_occ.name] = winning_occ

        if len(history) > 1:
            versions = [h.version_spec for h in history if h.version_spec]
            if len(set(versions)) > 1:
                conflict_warnings.append(
                    DiagnosticEvent(
                        type="conflicting_pin",
                        detail=f"Conflicting Explicit Pins for '{winning_occ.name}': Resolving to '{winning_occ.name}{winning_occ.version_spec}' ({time_qualifier}).",
                        cell_idx=winning_occ.cell_idx,
                        line_idx=winning_occ.line_idx,
                        level="warning"
                    )
                )

            flags_history = [tuple(h.flags) for h in history]
            if len(set(flags_history)) > 1:
                flags_display = " ".join(winning_occ.flags) if winning_occ.flags else "default index (no flags)"
                conflict_warnings.append(
                    DiagnosticEvent(
                        type="conflicting_flags",
                        detail=f"Conflicting Scoped Flags for '{winning_occ.name}': Overwriting earlier flags with '{flags_display}' ({time_qualifier}).",
                        cell_idx=winning_occ.cell_idx,
                        line_idx=winning_occ.line_idx,
                        level="warning"
                    )
                )

    return resolved, conflict_warnings


def harvest_scoped_cell_flags(code_sources: List[str]) -> Dict[str, List[str]]:
    """Convenience delegate returning harvested scoped flags map directly."""
    occurrences = harvest_pip_install_occurrences(code_sources)
    resolved, _ = resolve_pip_occurrences(occurrences)
    return {pkg: occ.flags for pkg, occ in resolved.items()}


def harvest_index_urls_from_sources(code_sources: List[str]) -> Set[str]:
    """Scans code sources for index URLs and returns a combined set of all harvested URLs."""
    h_res = harvest_cell_magics_and_commands(code_sources)
    return h_res.base_index_urls.union(h_res.extra_index_urls)


def harvest_cell_magics_and_commands(
    code_sources: List[str]
) -> HarvestResult:
    """Scans code sources for cell magics, index URLs, auxiliary tools, and shell commands."""
    occurrences = harvest_pip_install_occurrences(code_sources)
    resolved_occs, magic_warnings = resolve_pip_occurrences(occurrences)

    harvested_packages: Set[str] = set()
    base_index_urls: Set[str] = set()
    extra_index_urls: Set[str] = set()
    magic_notices: List[DiagnosticEvent] = []
    scoped_flags: Dict[str, List[str]] = {}

    for occ in occurrences:
        harvested_packages.add(occ.name)

    for pkg_key, occ in resolved_occs.items():
        scoped_flags[occ.name] = occ.flags
        i = 0
        while i < len(occ.flags):
            flag = occ.flags[i]
            val = occ.flags[i+1] if i + 1 < len(occ.flags) else ""
            if flag in {"--index-url", "-i"}:
                base_index_urls.add(val)
            elif flag in {"--extra-index-url", "-f", "--find-links"}:
                extra_index_urls.add(val)
            i += 2

    for cell_idx, source in enumerate(code_sources, start=1):
        cell_type, clean_body = classify_cell_source(source)
        if cell_type == "WRITEFILE":
            continue

        for line_idx, line in enumerate(clean_body.splitlines()):
            clean_line = line.strip()
            if not clean_line or clean_line.startswith('#') or clean_line in SHELL_CELL_MAGICS:
                continue

            command_segments = SHELL_SPLIT_PATTERN.split(clean_line)
            for segment in command_segments:
                seg = segment.strip()
                if not seg:
                    continue

                if SYSTEM_PKG_PATTERN.match(seg):
                    magic_notices.append(
                        DiagnosticEvent(
                            type="system_command",
                            detail=f"Cell {cell_idx} uses a system install command ('{seg}'). Note: System dependencies must be run manually by readers.",
                            cell_idx=cell_idx - 1,
                            line_idx=line_idx,
                            level="notice"
                        )
                    )
                elif CONDA_INSTALL_PATTERN.match(seg):
                    magic_notices.append(
                        DiagnosticEvent(
                            type="conda_command",
                            detail=f"Cell {cell_idx} uses 'conda install'. Conda packages are not tracked in pip requirements manifests.",
                            cell_idx=cell_idx - 1,
                            line_idx=line_idx,
                            level="notice"
                        )
                    )
                elif PIP_INSTALL_PATTERN.match(seg):
                    if "-r " in seg or "--requirement" in seg:
                        magic_warnings.append(
                            DiagnosticEvent(
                                type="external_requirement",
                                detail=f"Cell {cell_idx} references an external requirements file ('{seg}'). Ensure that file is shared alongside your notebook.",
                                cell_idx=cell_idx - 1,
                                line_idx=line_idx,
                                level="warning"
                            )
                        )

    return HarvestResult(
        harvested_packages=harvested_packages,
        base_index_urls=base_index_urls,
        extra_index_urls=extra_index_urls,
        magic_warnings=magic_warnings,
        magic_notices=magic_notices,
        scoped_flags=scoped_flags
    )


# =====================================================================
# UNIFIED TIMELINE ENGINE
# =====================================================================

def build_unified_timeline(
    code_sources: List[str],
    frozen_env: Dict[str, str],
    pkg_dist_map: Optional[Dict[str, List[str]]] = None,
    is_execution_ordered: bool = True,
    local_repo_modules: Optional[Set[str]] = None
) -> TimelineResult:
    """
    Constructs the master sequence of DependencyEntry objects:
    - Explicit pip install occurrences anchor timeline coordinates.
    - Bare AST imports only anchor position if no explicit install was found anywhere in the notebook.
    Returns a structured TimelineResult payload.
    """
    pip_occs = harvest_pip_install_occurrences(code_sources)
    resolved_pips, conflict_warnings = resolve_pip_occurrences(pip_occs, is_execution_ordered=is_execution_ordered)

    all_import_occs: List[ImportOccurrence] = []
    submodules_map: Dict[str, Set[str]] = {}
    guarded_set: Set[str] = set()

    for cell_idx, src in enumerate(code_sources):
        cell_imports = extract_import_occurrences_from_source(src, cell_idx=cell_idx)
        for imp in cell_imports:
            all_import_occs.append(imp)
            if imp.full_name and '.' in imp.full_name:
                submodules_map.setdefault(imp.module, set()).add(imp.full_name)
            if imp.is_guarded:
                guarded_set.add(imp.module)

    timeline_events: List[Tuple[Tuple[int, int], str, str]] = []
    seen_packages: Set[str] = set()

    # 1. Place explicit pip installs
    for norm_key, occ in resolved_pips.items():
        canon = canonicalize_pkg_name(occ.name)
        if norm_key == canon:
            coord = (occ.cell_idx, occ.line_idx)
            timeline_events.append((coord, "PIP", occ.name))
            seen_packages.add(canon)

    # 2. Place bare imports only if not already placed via pip install
    for imp in all_import_occs:
        norm_imp = canonicalize_pkg_name(imp.module)
        if imp.module.lower() in STD_LIB:
            continue
        pypi_name = IMPORT_TO_PYPI_MAP.get(imp.module, imp.module)
        canon_pypi = canonicalize_pkg_name(pypi_name)
        if norm_imp not in seen_packages and canon_pypi not in seen_packages:
            coord = (imp.cell_idx, imp.line_idx)
            timeline_events.append((coord, "IMPORT", imp.module))
            seen_packages.add(norm_imp)
            seen_packages.add(canon_pypi)

    timeline_events.sort(key=lambda t: t[0])

    dependencies: List[DependencyEntry] = []
    promotion_notices: List[PromotionDetail] = []

    for coord, kind, pkg_name in timeline_events:
        canon_name = canonicalize_pkg_name(pkg_name)
        submods = submodules_map.get(pkg_name, set())
        is_guarded = pkg_name in guarded_set

        if kind == "PIP":
            occ = resolved_pips[canon_name]
            dep_entry, promo = resolve_pypi_package_and_extras(
                occ.name, submods, frozen_env, pkg_dist_map=pkg_dist_map, is_guarded=is_guarded, local_repo_modules=local_repo_modules
            )
            dep_entry.source = "pip_command"
            if occ.version_spec and not dep_entry.is_comment:
                v_clean = occ.version_spec.lstrip("=<>!~")
                dep_entry.version = v_clean
                
                host_match = frozen_env.get(canon_name)
                if host_match and "==" in host_match:
                    host_ver = host_match.split("==", 1)[1]
                    if host_ver != v_clean:
                        logger.debug(
                            f"[Timeline] Explicit notebook pin '{pkg_name}=={v_clean}' preferred over active host version '{host_ver}'."
                        )
            dep_entry.flags = list(occ.flags)
            dependencies.append(dep_entry)
            if promo and promo not in promotion_notices:
                promotion_notices.append(promo)
        else:
            dep_entry, promo = resolve_pypi_package_and_extras(
                pkg_name, submods, frozen_env, pkg_dist_map=pkg_dist_map, is_guarded=is_guarded, local_repo_modules=local_repo_modules
            )
            dep_entry.source = "import"
            dependencies.append(dep_entry)
            if promo and promo not in promotion_notices:
                promotion_notices.append(promo)

    return TimelineResult(
        dependencies=dependencies,
        promotion_notices=promotion_notices,
        conflict_warnings=conflict_warnings
    )


# =====================================================================
# ENVIRONMENT CORRELATION & EXTRAS PROMOTION
# =====================================================================

def build_auxiliary_tool_entries(
    harvested_packages: Set[str],
    imported_packages: Any,
    frozen_env: Dict[str, str]
) -> List[DependencyEntry]:
    """Builds commented DependencyEntry instances for CLI tools installed via cell magics."""
    aux_entries: List[DependencyEntry] = []
    imported_set = {canonicalize_pkg_name(imp) for imp in imported_packages}
    unimported_tools = sorted([
        pkg for pkg in harvested_packages 
        if canonicalize_pkg_name(pkg) not in imported_set and pkg.lower() not in STD_LIB
    ])

    if not unimported_tools:
        return aux_entries

    aux_entries.append(DependencyEntry(
        is_comment=True,
        source="pip_command",
        status="auxiliary_tool",
        comment_text="\n# --- AUXILIARY TOOL INSTALLS (harvested from cell magics) ---"
    ))
    for tool in unimported_tools:
        canon_tool = canonicalize_pkg_name(tool)
        matched_pin = frozen_env.get(canon_tool)
        ver = matched_pin.split("==", 1)[1] if matched_pin and "==" in matched_pin else ""
        if matched_pin:
            aux_entries.append(DependencyEntry(
                name=tool,
                version=ver,
                source="pip_command",
                status="auxiliary_tool",
                is_comment=True,
                comment_text=f"# {matched_pin}  (installed via cell command; not directly imported in Python code)"
            ))
        else:
            aux_entries.append(DependencyEntry(
                name=tool,
                version="",
                source="pip_command",
                status="auxiliary_tool",
                is_comment=True,
                comment_text=f"# {tool}  (installed via cell command; not found in active env)"
            ))

    return aux_entries


def build_writefile_tool_entries(
    writefile_imports: Any,
    primary_imports: Any,
    frozen_env: Dict[str, str]
) -> List[DependencyEntry]:
    """Builds commented DependencyEntry instances for dependencies inside %%writefile scripts."""
    entries: List[DependencyEntry] = []
    primary_set = {canonicalize_pkg_name(imp) for imp in primary_imports}
    script_only = sorted([
        pkg for pkg in writefile_imports 
        if canonicalize_pkg_name(pkg) not in primary_set and pkg.lower() not in STD_LIB
    ])

    if not script_only:
        return entries

    entries.append(DependencyEntry(
        is_comment=True,
        source="writefile_script",
        status="writefile_script",
        comment_text="\n# --- WRITEFILE SCRIPT DEPENDENCIES ---"
    ))
    for pkg in script_only:
        pypi_name = IMPORT_TO_PYPI_MAP.get(pkg, pkg)
        canon_pypi = canonicalize_pkg_name(pypi_name)
        matched_pin = frozen_env.get(canon_pypi)
        ver = matched_pin.split("==", 1)[1] if matched_pin and "==" in matched_pin else ""
        if matched_pin:
            entries.append(DependencyEntry(
                name=pypi_name,
                version=ver,
                source="writefile_script",
                status="writefile_script",
                is_comment=True,
                comment_text=f"# {matched_pin}  (imported inside script generated via %%writefile)"
            ))
        else:
            entries.append(DependencyEntry(
                name=pypi_name,
                version="",
                source="writefile_script",
                status="writefile_script",
                is_comment=True,
                comment_text=f"# {pypi_name}  (imported inside script generated via %%writefile; not found in active env)"
            ))

    return entries


def resolve_pypi_package_and_extras(
    imp: str, 
    submodules_set: Set[str], 
    frozen_env: Dict[str, str], 
    pkg_dist_map: Optional[Dict[str, List[str]]] = None,
    is_guarded: bool = False,
    local_repo_modules: Optional[Set[str]] = None
) -> Tuple[DependencyEntry, Optional[PromotionDetail]]:
    """Resolves top-level import to a DependencyEntry."""
    if imp in PLATFORM_PSEUDO_MODULES:
        return DependencyEntry(
            name=imp,
            status="platform_pseudo_module",
            is_comment=True,
            comment_text=f"# {imp} (provided automatically by platform like Colab/Databricks; no install needed)"
        ), None

    if imp in BUILD_AND_PACKAGING_TOOLS:
        return DependencyEntry(
            name=imp,
            status="build_tool",
            is_comment=True,
            comment_text=f"# {imp} (core Python build/packaging tool; excluded from requirement lockfiles)"
        ), None

    if local_repo_modules and imp in local_repo_modules:
        return DependencyEntry(
            name=imp,
            status="local_module",
            is_comment=True,
            comment_text=f"# {imp} (local folder/file next to notebook; ensure sibling files were shared)"
        ), None

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

    canon_pypi = canonicalize_pkg_name(pypi_name)
    matched_pin = frozen_env.get(canon_pypi)

    if is_guarded:
        if matched_pin:
            ver = matched_pin.split("==", 1)[1]
            return DependencyEntry(
                name=pypi_name,
                version=ver,
                status="guarded",
                is_comment=True,
                comment_text=f"# {matched_pin} (optional or conditional dependency inside try/except block)"
            ), None
        return DependencyEntry(
            name=pypi_name,
            version="",
            status="guarded",
            is_comment=True,
            comment_text=f"# {pypi_name} (optional or conditional dependency inside try/except block)"
        ), None

    if not matched_pin:
        return DependencyEntry(
            name=pypi_name,
            version="",
            status="pinned",
            is_comment=True,
            comment_text=f"# {pypi_name} (imported as '{imp}', not currently found in active env)"
        ), None

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
        promoted_name = f"{pkg_part}[{extra_tag}]"
        promoted_pin = f"{promoted_name}=={ver_part}"
        notice_detail = f"💡 Extra Dependency Promotion: importing '{imp}.{extra_tag}' automatically promoted requirement to '{promoted_pin}'"
        promo = PromotionDetail(
            import_name=f"{imp}.{extra_tag}",
            promoted_name=promoted_name,
            version=ver_part,
            detail=notice_detail
        )
        return DependencyEntry(name=promoted_name, version=ver_part, status="pinned"), promo

    return DependencyEntry(name=pkg_part, version=ver_part, status="pinned"), None


@_memoize_for_run
def build_manifest_entries(
    imports: Any, 
    submodules: Dict[str, Set[str]], 
    frozen_env: Dict[str, str], 
    pkg_dist_map: Optional[Dict[str, List[str]]] = None,
    guarded_imports: Optional[Set[str]] = None,
    local_repo_modules: Optional[Set[str]] = None
) -> Tuple[List[str], List[str]]:
    """Builds string-formatted manifest lines for legacy/batch consumers while preserving order."""
    entries, promotions = build_dependency_objects(
        imports, submodules, frozen_env, pkg_dist_map, guarded_imports, local_repo_modules
    )
    pinned_manifest = [e.specifier for e in entries]
    notices = [p.detail for p in promotions if p.detail]
    return pinned_manifest, notices


def build_dependency_objects(
    imports: Any, 
    submodules: Dict[str, Set[str]], 
    frozen_env: Dict[str, str], 
    pkg_dist_map: Optional[Dict[str, List[str]]] = None,
    guarded_imports: Optional[Set[str]] = None,
    local_repo_modules: Optional[Set[str]] = None
) -> Tuple[List[DependencyEntry], List[PromotionDetail]]:
    """Generates typed DependencyEntry instances in first-encountered order."""
    entries: List[DependencyEntry] = []
    promotions: List[PromotionDetail] = []
    guarded_set = guarded_imports or set()

    for imp in imports:
        if imp in STD_LIB:
            continue
        submods = submodules.get(imp, set())
        is_guarded = imp in guarded_set
        dep_entry, promo = resolve_pypi_package_and_extras(
            imp, submods, frozen_env, pkg_dist_map=pkg_dist_map, is_guarded=is_guarded, local_repo_modules=local_repo_modules
        )
        entries.append(dep_entry)
        if promo and promo not in promotions:
            promotions.append(promo)

    return entries, promotions


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
    if res.returncode != 0:
        logger.warning(f"⚠️ 'pip freeze' execution failed (exit code {res.returncode}). Active environment versions could not be captured.")
        return {}, []

    frozen: Dict[str, str] = {}
    for line in res.stdout.splitlines():
        if "==" in line:
            pkg, ver = line.split("==", 1)
            canon = canonicalize_pkg_name(pkg)
            frozen[canon] = line.strip()
            frozen[pkg.lower()] = line.strip()
            
    return frozen, res.stdout.splitlines()


def process_package_requirements(
    pinned_list: List[str], 
    harvested_urls: Set[str],
    base_urls: Optional[Set[str]] = None,
    auxiliary_entries: Optional[List[str]] = None,
    writefile_entries: Optional[List[str]] = None
) -> Tuple[List[str], List[Tuple[str, List[str]]], List[str]]:
    """Legacy compatibility helper: correlates pinned packages with index URLs and auxiliary entries."""
    manifest_output: List[str] = []
    local_tagged_info: List[Tuple[str, List[str]]] = []
    warnings_out: List[str] = []
    
    if base_urls:
        for url in sorted(base_urls):
            manifest_output.append(f"--index-url {url}")

    extra_urls = harvested_urls - (base_urls or set())
    if extra_urls:
        for url in sorted(extra_urls):
            manifest_output.append(f"--extra-index-url {url}")

    for item in pinned_list:
        manifest_output.append(item)
        if '+' in item:
            all_urls = sorted(harvested_urls.union(base_urls or set()))
            local_tagged_info.append((item, all_urls))
            if not all_urls:
                warnings_out.append(item)

    if auxiliary_entries:
        manifest_output.extend(auxiliary_entries)

    if writefile_entries:
        manifest_output.extend(writefile_entries)
            
    return manifest_output, local_tagged_info, warnings_out


def build_dependency_entries(
    dependencies: List[DependencyEntry],
    scoped_flags: Optional[Dict[str, List[str]]] = None,
    auxiliary_entries: Optional[List[DependencyEntry]] = None,
    writefile_entries: Optional[List[DependencyEntry]] = None
) -> Tuple[List[DependencyEntry], List[Tuple[str, List[str]]], List[DiagnosticEvent]]:
    """Attaches scoped flags to DependencyEntry objects and identifies local hardware tags."""
    flags_map = scoped_flags or {}
    local_tagged_info: List[Tuple[str, List[str]]] = []
    warnings_out: List[DiagnosticEvent] = []
    all_entries: List[DependencyEntry] = []

    for dep in dependencies:
        if not dep.is_comment and dep.name:
            matched_flags: List[str] = []
            canon_name = canonicalize_pkg_name(dep.name)
            for candidate in (dep.name, dep.name.lower(), canon_name):
                if candidate in flags_map:
                    matched_flags = flags_map[candidate]
                    break
            if not dep.flags:
                dep.flags = matched_flags

            if '+' in dep.version:
                local_tagged_info.append((dep.specifier, dep.flags))
                if not dep.flags:
                    warnings_out.append(
                        DiagnosticEvent(
                            type="missing_hardware_index",
                            detail=f"Specific hardware build detected: `{dep.specifier}` with no download URL harvested in code cells.",
                            level="warning"
                        )
                    )

        all_entries.append(dep)

    if auxiliary_entries:
        all_entries.extend(auxiliary_entries)

    if writefile_entries:
        all_entries.extend(writefile_entries)

    return all_entries, local_tagged_info, warnings_out


# =====================================================================
# HARDWARE ACCELERATION INSPECTION
# =====================================================================

def expand_transitive_frameworks(imports: Any) -> Set[str]:
    """Expands a set or list of import stems to include their base GPU framework."""
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
    """Probes PyTorch for CUDA or Apple Silicon MPS acceleration."""
    try:
        import torch
    except ImportError:
        return None
    except Exception as e:
        logger.debug(f"[HardwareProbe] PyTorch import failed: {e}")
        raise

    if torch.cuda.is_available():
        dev_name = f"{torch.cuda.get_device_name(0)} (via PyTorch)"
        return GpuProbeResult("NVIDIA CUDA", dev_name)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return GpuProbeResult("Apple Silicon MPS", "Apple Silicon GPU (Metal via PyTorch)")
    return None


def probe_tensorflow_gpu() -> Optional[GpuProbeResult]:
    """Probes TensorFlow for GPU acceleration while silencing C++ CUDA driver noise."""
    try:
        with silence_fd2_stderr():
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if not gpus:
                return None
            dev_name = "NVIDIA GPU (via TensorFlow)"
            try:
                details = tf.config.experimental.get_device_details(gpus[0])
                dev_name = f"{details.get('device_name', 'NVIDIA GPU')} (via TensorFlow)"
            except Exception:
                pass
            return GpuProbeResult("GPU", dev_name)
    except ImportError:
        return None
    except Exception as e:
        logger.debug(f"[HardwareProbe] TensorFlow GPU probe failed unexpectedly: {e}")
        raise


def probe_jax_gpu() -> Optional[GpuProbeResult]:
    """Probes JAX for GPU/TPU acceleration while silencing C++ CUDA driver noise."""
    try:
        with silence_fd2_stderr():
            import jax
            accelerators = [d for d in jax.devices() if d.platform.lower() in ("gpu", "tpu", "metal")]
            if not accelerators:
                return None
            first_accel = accelerators[0]
            accel_type = first_accel.platform.upper()
            dev_name = f"{accel_type} ({first_accel.device_kind}) via JAX"
            return GpuProbeResult(accel_type, dev_name)
    except ImportError:
        return None
    except Exception as e:
        logger.debug(f"[HardwareProbe] JAX GPU probe failed unexpectedly: {e}")
        raise


GPU_PROBES: List[Tuple[str, Callable[[], Optional[GpuProbeResult]]]] = [
    ("torch", probe_torch_gpu),
    ("tensorflow", probe_tensorflow_gpu),
    ("jax", probe_jax_gpu),
]


def inspect_gpu_environment(imported_packages: Any) -> Optional[GpuInfo]:
    """Coordinates per-framework GPU/accelerator probing across PyTorch, TensorFlow, and JAX."""
    expanded_imports = expand_transitive_frameworks(imported_packages)
    found_frameworks = list(SUPPORTED_GPU_FRAMEWORKS.intersection(expanded_imports))
    if not found_frameworks:
        return None

    framework_devices: Dict[str, Optional[str]] = {}
    active_types: List[str] = []
    probe_errors: List[str] = []
    primary_fw: Optional[str] = None
    primary_dev: Optional[str] = None

    for fw_stem, probe in GPU_PROBES:
        if fw_stem not in found_frameworks:
            continue
        try:
            result = probe()
            if result:
                framework_devices[fw_stem] = result.device_name
                active_types.append(result.accelerator_type)
                if not primary_dev:
                    primary_fw = CANONICAL_TO_FRAMEWORK_DISPLAY.get(fw_stem, fw_stem.capitalize())
                    primary_dev = result.device_name
            else:
                framework_devices[fw_stem] = None
        except Exception as e:
            framework_devices[fw_stem] = None
            fw_label = CANONICAL_TO_FRAMEWORK_DISPLAY.get(fw_stem, fw_stem.capitalize())
            probe_errors.append(f"{fw_label} probe error: {e}")

    has_gpu = primary_dev is not None

    return GpuInfo(
        has_gpu=has_gpu,
        type=active_types[0] if active_types else None,
        active_framework=primary_fw,
        device_name=primary_dev,
        frameworks=sorted(found_frameworks),
        framework_devices=framework_devices,
        probe_errors=probe_errors
    )


def resolve_notebook_gpu_info(nb_imports: Any, batch_hw_cache: Optional[GpuInfo]) -> Optional[GpuInfo]:
    """Matches a notebook's specific imports against the active batch hardware cache."""
    if not batch_hw_cache:
        return None

    expanded_nb_imports = expand_transitive_frameworks(nb_imports)
    nb_fw = set(batch_hw_cache.frameworks).intersection(expanded_nb_imports)
    if not nb_fw:
        return None

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
        return GpuInfo(
            has_gpu=True,
            type=batch_hw_cache.type,
            active_framework=active_label,
            device_name=matched_device,
            frameworks=sorted(nb_fw),
            framework_devices=fw_devices
        )
    else:
        return GpuInfo(
            has_gpu=False,
            type=None,
            active_framework=None,
            device_name=None,
            frameworks=sorted(nb_fw),
            framework_devices=fw_devices
        )


# =====================================================================
# BLUEPRINT & SEQUENTIAL INSTALL SETUP GENERATOR
# =====================================================================

def generate_production_blueprint(
    manifest_items: List[Union[DependencyEntry, Dict[str, Any], str]], 
    full_freeze_lines: Optional[List[str]] = None, 
    local_tagged_info: Optional[List[Tuple[str, List[str]]]] = None, 
    gpu_info: Optional[GpuInfo] = None
) -> BlueprintResult:
    """Assembles Cell 1 Markdown and Cell 2 Python code using structured DependencyEntry objects."""
    py_major, py_minor = sys.version_info.major, sys.version_info.minor
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    normalized_items: List[Dict[str, Any]] = []
    comment_lines: List[str] = []

    for item in manifest_items:
        if isinstance(item, DependencyEntry):
            if item.is_comment:
                comment_lines.append(item.comment_text)
            else:
                normalized_items.append(item.to_dict())
        elif isinstance(item, dict):
            normalized_items.append(item)
        elif isinstance(item, str):
            clean_item = item.strip()
            if clean_item.startswith("#") or clean_item.startswith("--"):
                comment_lines.append(clean_item)
                continue
            parts = clean_item.split("==")
            name = parts[0]
            ver = parts[1] if len(parts) > 1 else ""
            normalized_items.append({"name": name, "version": ver, "flags": []})

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
        f"This notebook includes verified dependencies to ensure reproducible execution.\n",
        "- **Automatic Setup:** Cell 2 verifies Python version compatibility and installs verified package versions sequentially."
    ]
    
    if gpu_markdown_section:
        markdown_lines.append(gpu_markdown_section)
    if local_builds_section:
        markdown_lines.append(local_builds_section)
        
    markdown_lines.append("- **Network Notice:** Active internet access is required to download uncached packages.")

    step1_markdown = "\n".join(markdown_lines)
    
    comments_block = ""
    if comment_lines:
        comments_block = "\n# Informational notes & uninstalled fallbacks:\n" + "\n".join(comment_lines) + "\n"

    freeze_block_code = ""
    if full_freeze_lines:
        freeze_lines_repr = repr(full_freeze_lines)
        freeze_block_code = f"\n# --- FULL FREEZE FALLBACK BLOCK ---\nFULL_FREEZE_FALLBACK = {freeze_lines_repr}\n"

    step2_code = f"""# =====================================================================
# VERIFIED ENVIRONMENT DEPENDENCIES ({timestamp})
# =====================================================================

import sys
import subprocess
import importlib.metadata

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

# Dependency Specification with Scoped Flags
DEPENDENCIES = {repr(normalized_items)}
{comments_block}{freeze_block_code}
print(f"Applying verified environment dependencies [{timestamp}]...")
print("💡 Note: Dependencies are installed sequentially to prevent index conflicts.\\n")

passed_count = 0
failed_packages = []
total_deps = len(DEPENDENCIES)
installed_baseline = {{}}

for idx, item in enumerate(DEPENDENCIES, start=1):
    name = item["name"]
    ver = item.get("version", "")
    flags = item.get("flags", [])
    specifier = f"{{name}}=={{ver}}" if ver else name

    cmd = [sys.executable, "-m", "pip", "install", specifier] + flags
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        passed_count += 1
        print(f"[{{idx}}/{{total_deps}}] ✅ {{specifier}} installed successfully")
        
        # Real-time drift audit across previously installed dependencies
        try:
            current_ver = importlib.metadata.version(name)
            installed_baseline[name] = current_ver
        except Exception:
            pass

        for prev_pkg, prev_ver in list(installed_baseline.items()):
            if prev_pkg == name:
                continue
            try:
                active_now = importlib.metadata.version(prev_pkg)
                if active_now != prev_ver:
                    print(f"   ⚠️ Dependency Drift: Installing '{{specifier}}' caused '{{prev_pkg}}' to drift from {{prev_ver}} ➔ {{active_now}}")
                    installed_baseline[prev_pkg] = active_now
            except Exception:
                pass
    else:
        failed_packages.append((specifier, ver, flags, result.stderr))
        print(f"[{{idx}}/{{total_deps}}] ❌ {{specifier}} failed to install")
        print(f"   ├─ Author Verified Version: {{ver or 'unspecified'}}")
        if flags:
            print(f"   ├─ Scoped Flags: {{' '.join(flags)}}")
        err_snippet = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "Unknown pip error"
        print(f"   └─ Error: {{err_snippet}}\\n")

print("\\n" + "=" * 60)
if not failed_packages:
    print(f"✅ Setup complete! All {{passed_count}}/{{total_deps}} dependencies verified.")
else:
    print(f"⚠️ Setup completed with issues: {{passed_count}}/{{total_deps}} packages installed.")
    print("Troubleshooting Steps:")
    print("1. Internet Access: Ensure your notebook environment has active internet access.")
    print("2. Unpinned Installs: Test installing failed libraries manually: '!pip install <pkg>'")
    print(f"3. Troubleshooting Steps: For a detailed guide on resolving setup errors, see: {HELP_URL}")
print("=" * 60)"""

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
# =====================================================================

class RepoEnvironmentMap:
    """Aggregates notebook scan results across a repository directory."""
    def __init__(self, target_dir: str) -> None:
        self.target_dir = target_dir
        self.scan_results: List[NotebookScanResult] = []
        self.non_python_files: List[NotebookScanResult] = []
        self.parse_errors: List[NotebookScanResult] = []
        self.companion_files_skipped: List[Path] = []
        self.global_imports: List[str] = []
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
                if imp not in self.global_imports:
                    self.global_imports.append(imp)
                self.package_to_notebooks.setdefault(imp, []).append(result.path)

        for pkg in result.harvested_pkgs:
            if pkg not in STD_LIB:
                if pkg not in self.global_imports:
                    self.global_imports.append(pkg)
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


def walk_and_scan_directory(target_dir: str, skip_suffix: Optional[str] = None) -> RepoEnvironmentMap:
    """Recursively scans directory for .ipynb files in batch mode."""
    repo_map = RepoEnvironmentMap(target_dir)
    target_path = Path(target_dir)

    for root, dirs, files in os.walk(target_path):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in DEFAULT_IGNORED_DIRS]
        for file in sorted(files):
            if file.endswith('.ipynb'):
                full_path = Path(root) / file

                if skip_suffix and full_path.stem.endswith(skip_suffix):
                    repo_map.companion_files_skipped.append(full_path)
                    continue

                ext_res = extract_from_file(str(full_path), strict=True)
                
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
                    writefile_imports=ext_res.writefile_imports,
                    harvested_pkgs=h_res.harvested_packages,
                    base_index_urls=h_res.base_index_urls,
                    extra_index_urls=h_res.extra_index_urls,
                    scoped_flags=h_res.scoped_flags,
                    magic_warnings=h_res.magic_warnings,
                    magic_notices=h_res.magic_notices
                )
                repo_map.add_result(res)

    return repo_map


def build_single_notebook_report(
    scan_res: NotebookScanResult,
    frozen_env: Dict[str, str],
    pkg_dist_map: Dict[str, List[str]],
    gpu_info: Optional[GpuInfo],
    local_repo_modules: Optional[Set[str]] = None
) -> NotebookAnalysisReport:
    """Builds a complete NotebookAnalysisReport object for a single notebook."""
    if local_repo_modules is None:
        local_repo_modules = get_notebook_local_modules(scan_res.path)

    timeline_res = build_unified_timeline(
        scan_res.code_sources,
        frozen_env=frozen_env,
        pkg_dist_map=pkg_dist_map,
        local_repo_modules=local_repo_modules
    )

    timeline_pkgs = {canonicalize_pkg_name(d.name) for d in timeline_res.dependencies if d.name}
    aux_entries = build_auxiliary_tool_entries(scan_res.harvested_pkgs - timeline_pkgs, scan_res.imports, frozen_env)    
    writefile_entries = build_writefile_tool_entries(scan_res.writefile_imports, scan_res.imports, frozen_env)

    all_dep_entries, _, hw_warnings = build_dependency_entries(
        timeline_res.dependencies,
        scoped_flags=scan_res.scoped_flags,
        auxiliary_entries=aux_entries,
        writefile_entries=writefile_entries
    )

    all_warnings: List[DiagnosticEvent] = []
    all_warnings.extend(scan_res.dynamic_warnings)
    all_warnings.extend(scan_res.magic_warnings)
    all_warnings.extend(timeline_res.conflict_warnings)
    all_warnings.extend(hw_warnings)

    local_mods_detected = sorted(list(local_repo_modules.intersection(set(scan_res.imports))))
    pseudo_mods_detected = sorted(list(PLATFORM_PSEUDO_MODULES.intersection(set(scan_res.imports))))
    build_tools_detected = sorted(list(BUILD_AND_PACKAGING_TOOLS.intersection(set(scan_res.imports))))

    return NotebookAnalysisReport(
        notebook_path=str(scan_res.path),
        is_python=scan_res.is_python,
        lang_label=scan_res.lang_label,
        parse_error=scan_res.parse_error,
        dependencies=all_dep_entries,
        local_modules=local_mods_detected,
        platform_pseudo_modules=pseudo_mods_detected,
        build_and_packaging_tools=build_tools_detected,
        gpu=gpu_info,
        warnings=all_warnings,
        notices=scan_res.magic_notices,
        promotions=timeline_res.promotion_notices
    )


def analyze_batch_repository(
    repo_map: RepoEnvironmentMap, 
    frozen_env: Dict[str, str], 
    pkg_dist_map: Dict[str, List[str]], 
    batch_hw_cache: Optional[GpuInfo]
) -> BatchAnalysisSummary:
    """Aggregates dependency metrics, warnings, and index settings across repository notebooks."""
    parse_errors_list = [
        {"path": str(err_res.path), "cause": err_res.parse_error or "Unknown parse error"}
        for err_res in repo_map.parse_errors
    ]

    summary = BatchAnalysisSummary(
        target_dir=repo_map.target_dir,
        total_python_notebooks=len(repo_map.scan_results),
        non_python_count=len(repo_map.non_python_files),
        companion_skipped_count=len(repo_map.companion_files_skipped),
        parse_errors=parse_errors_list,
        batch_hw_cache=batch_hw_cache
    )

    for item in repo_map.non_python_files:
        summary.non_python_languages[item.lang_label] = (
            summary.non_python_languages.get(item.lang_label, 0) + 1
        )

    canonical_to_display: Dict[str, str] = {}
    canonical_missing_map: Dict[str, List[str]] = {}

    for res in repo_map.scan_results:
        nb_local_mods = get_notebook_local_modules(res.path, repo_map.target_dir)
        nb_gpu_info = resolve_notebook_gpu_info(res.imports, batch_hw_cache)

        nb_report = build_single_notebook_report(
            res, frozen_env, pkg_dist_map, nb_gpu_info, local_repo_modules=nb_local_mods
        )
        summary.notebooks.append(nb_report)

        for dep in nb_report.dependencies:
            if not dep.is_comment and dep.version and "+" in dep.version:
                if not dep.flags:
                    summary.batch_hardware_warnings.setdefault(dep.specifier, []).append(Path(nb_report.notebook_path).name)

            if dep.is_comment:
                if dep.status in {"platform_pseudo_module", "build_tool", "local_module"}:
                    continue
                pypi_name = dep.name or (dep.comment_text.split()[1] if len(dep.comment_text.split()) > 1 else "")
                if pypi_name:
                    canon = canonicalize_pkg_name(pypi_name)
                    canonical_missing_map.setdefault(canon, []).append(Path(nb_report.notebook_path).name)
                    display_name = pypi_name.replace("_", "-")
                    canonical_to_display.setdefault(canon, display_name)
            elif dep.name:
                summary.matched_packages.add(dep.name.split("[")[0])

        for promo in nb_report.promotions:
            if promo not in summary.promotions:
                summary.promotions.append(promo)

        for warn in res.dynamic_warnings:
            if warn not in summary.dynamic_warnings:
                summary.dynamic_warnings.append(warn)

        for warn in res.magic_warnings:
            if warn not in summary.magic_warnings:
                summary.magic_warnings.append(warn)

        for notice in res.magic_notices:
            if notice not in summary.magic_notices:
                summary.magic_notices.append(notice)

    for canon, nbs in canonical_missing_map.items():
        disp_name = canonical_to_display.get(canon, canon)
        summary.missing_packages[disp_name] = sorted(list(set(nbs)))

    primary_url, url_reason = select_primary_index_url(repo_map.url_to_notebooks)
    summary.primary_url = primary_url
    summary.primary_url_reason = url_reason

    return summary


def format_console_report(summary: BatchAnalysisSummary) -> str:
    """Formats a BatchAnalysisSummary into a human-readable stdout report string."""
    out = []
    out.append("=" * 80)
    out.append("REPOSITORY REPRODUCIBILITY SUMMARY")
    out.append(f"Target Directory: {summary.target_dir}")
    out.append(f"Active Interpreter: {sys.executable}")
    out.append("=" * 80 + "\n")

    out.append("📁 NOTEBOOK INVENTORY & LANGUAGE SCAN:")
    out.append(f"  • Python (.ipynb): {summary.total_python_notebooks} files analyzed")
    
    if summary.companion_skipped_count > 0:
        out.append(f"  • Companion outputs skipped: {summary.companion_skipped_count} files")

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
        for err_dict in summary.parse_errors:
            out.append(f"  • {err_dict['path']}")
            out.append(f"    └─ Cause: {err_dict['cause']}")
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
            out.append(f"  • {warn.format_console()}")
        for warn in summary.magic_warnings:
            out.append(f"  • {warn.format_console()}")
        out.append("")

    if summary.magic_notices:
        out.append("ℹ️ SYSTEM & CONDA COMMANDS:")
        for notice in summary.magic_notices:
            out.append(f"  • {notice.format_console()}")
        out.append("")

    if summary.promotions:
        out.append("💡 AUTOMATIC EXTRA PROMOTIONS:")
        for promo in summary.promotions:
            out.append(f"  • {promo.detail}")
        out.append("")

    out.append("⚡ ACCELERATOR & DOWNLOAD INDEX CHECK:")
    if summary.batch_hw_cache and summary.batch_hw_cache.has_gpu:
        out.append(f"  • Active Hardware Accelerator: {summary.batch_hw_cache.device_name}")
    elif summary.batch_hw_cache and summary.batch_hw_cache.probe_errors:
        err_msg = "; ".join(summary.batch_hw_cache.probe_errors)
        out.append(f"  • Active Hardware Accelerator: None detected (⚠️ Detection encountered errors: {err_msg})")
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


def format_batch_report(summary: BatchAnalysisSummary) -> str:
    """Legacy alias redirecting to format_console_report."""
    return format_console_report(summary)


def format_json_batch_report(summary: BatchAnalysisSummary, artifacts_written: Optional[Dict[str, Any]] = None) -> str:
    """Formats a BatchAnalysisSummary into valid machine-readable JSON."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "mode": "batch",
        "target_dir": summary.target_dir,
        "environment": {
            "active_interpreter": sys.executable,
            "python_version": [sys.version_info.major, sys.version_info.minor, sys.version_info.micro]
        },
        "summary": {
            "is_clean": summary.is_clean,
            "total_python_notebooks": summary.total_python_notebooks,
            "non_python_count": summary.non_python_count,
            "non_python_languages": summary.non_python_languages,
            "companion_skipped_count": summary.companion_skipped_count,
            "parse_errors": summary.parse_errors,
            "matched_packages": sorted(list(summary.matched_packages)),
            "missing_packages": summary.missing_packages,
            "hardware_warnings": summary.batch_hardware_warnings,
            "promotions": [p.to_dict() for p in summary.promotions],
            "primary_index_url": summary.primary_url,
            "primary_index_url_reason": summary.primary_url_reason
        },
        "notebooks": [nb.to_dict() for nb in summary.notebooks],
        "artifacts_written": artifacts_written
    }
    return json.dumps(payload, indent=2)


def format_json_single_report(nb_report: NotebookAnalysisReport, artifacts_written: Optional[Dict[str, Any]] = None) -> str:
    """Formats a single NotebookAnalysisReport into valid machine-readable JSON."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "mode": "single_file",
        "environment": {
            "active_interpreter": sys.executable,
            "python_version": [sys.version_info.major, sys.version_info.minor, sys.version_info.micro]
        },
        **nb_report.to_dict(),
        "artifacts_written": artifacts_written
    }
    return json.dumps(payload, indent=2)


def generate_batch_analysis_report(
    repo_map: RepoEnvironmentMap, 
    frozen_env: Dict[str, str], 
    pkg_dist_map: Dict[str, List[str]], 
    batch_hw_cache: Optional[GpuInfo]
) -> Tuple[str, bool]:
    """Orchestrates batch repository analysis and returns (report_text, is_clean)."""
    summary = analyze_batch_repository(repo_map, frozen_env, pkg_dist_map, batch_hw_cache)
    report_text = format_console_report(summary)
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
            if not aux.comment_text.startswith("\n# ---"):
                pinned_entries_set.add(aux.comment_text)

    for entry in sorted(pinned_entries_set):
        lines.append(entry)

    return "\n".join(lines)


def apply_output_to_notebook(
    scan_res: NotebookScanResult, 
    frozen_env: Dict[str, str], 
    pkg_dist_map: Dict[str, List[str]], 
    batch_hw_cache: Optional[GpuInfo], 
    suffix: Optional[str] = None, 
    in_place: bool = False,
    local_repo_modules: Optional[Set[str]] = None,
    root_dir: Optional[str] = None,
    output_dir: Optional[str] = None
) -> Path:
    """Writes per-notebook locked file or replaces setup cells in-place idempotently."""
    if local_repo_modules is None:
        local_repo_modules = get_notebook_local_modules(scan_res.path, root_dir)

    timeline_res = build_unified_timeline(
        scan_res.code_sources,
        frozen_env=frozen_env,
        pkg_dist_map=pkg_dist_map,
        local_repo_modules=local_repo_modules
    )

    timeline_pkgs = {canonicalize_pkg_name(d.name) for d in timeline_res.dependencies if d.name}
    aux_entries = build_auxiliary_tool_entries(scan_res.harvested_pkgs - timeline_pkgs, scan_res.imports, frozen_env)    
    writefile_entries = build_writefile_tool_entries(scan_res.writefile_imports, scan_res.imports, frozen_env)
    
    all_dep_entries, local_tagged, _ = build_dependency_entries(
        timeline_res.dependencies, 
        scoped_flags=scan_res.scoped_flags, 
        auxiliary_entries=aux_entries, 
        writefile_entries=writefile_entries
    )
    
    gpu_info = resolve_notebook_gpu_info(scan_res.imports, batch_hw_cache)

    blueprint = generate_production_blueprint(all_dep_entries, local_tagged_info=local_tagged, gpu_info=gpu_info)
    managed_cells = create_managed_cells(blueprint)

    with open(scan_res.path, 'r', encoding='utf-8') as f:
        nb_data = json.load(f)

    cells = nb_data.get("cells", [])

    # Idempotent filter: strip prior managed setup blocks
    non_managed_cells = [
        c for c in cells 
        if not (isinstance(c.get("metadata"), dict) and c.get("metadata", {}).get("notebook_env", {}).get("managed") is True)
    ]
    nb_data["cells"] = managed_cells + non_managed_cells

    if in_place:
        target_path = scan_res.path
    elif output_dir:
        out_base = Path(output_dir)
        stem = scan_res.path.stem
        active_suffix = suffix if suffix is not None else ""
        file_name = f"{stem}{active_suffix}.ipynb"

        if root_dir and Path(root_dir).exists():
            try:
                rel_parent = scan_res.path.parent.relative_to(Path(root_dir))
                dest_dir = out_base / rel_parent
            except ValueError:
                dest_dir = out_base
        else:
            dest_dir = out_base

        dest_dir.mkdir(parents=True, exist_ok=True)
        target_path = dest_dir / file_name
    else:
        stem = scan_res.path.stem
        active_suffix = suffix if suffix is not None else "_merged"
        target_path = scan_res.path.parent / f"{stem}{active_suffix}.ipynb"

    with open(target_path, 'w', encoding='utf-8') as f:
        json.dump(nb_data, f, indent=1)

    return target_path


def run_batch_pipeline(
    target_batch_dir: str, 
    args: argparse.Namespace, 
    frozen_env: Dict[str, str], 
    pkg_dist_map: Dict[str, List[str]], 
    batch_hw_cache: Optional[GpuInfo],
    precomputed_repo_map: Optional[RepoEnvironmentMap] = None
) -> None:
    """Executes the batch processing pipeline across a directory of notebooks."""
    effective_suffix = args.suffix if args.suffix is not None else "_merged"
    skip_suffix = None if args.in_place else effective_suffix
    repo_map = precomputed_repo_map or walk_and_scan_directory(target_batch_dir, skip_suffix=skip_suffix)
    summary = analyze_batch_repository(repo_map, frozen_env, pkg_dist_map, batch_hw_cache)

    is_json = getattr(args, "format", "text") == "json"
    if not is_json:
        print(format_console_report(summary))

    if not summary.is_clean and (args.universal or args.output or args.in_place or args.output_dir):
        logger.error("\n❌ Execution aborted: Resolve file/parse errors before running --universal, --output, --output-dir, or --in-place.")
        if is_json:
            print(format_json_batch_report(summary))
        sys.exit(1)

    artifacts_written: Dict[str, Any] = {}

    if args.universal:
        manifest_filename = args.universal if isinstance(args.universal, str) else DEFAULT_UNIVERSAL_MANIFEST_NAME
        uni_content = generate_universal_manifest(repo_map, frozen_env, pkg_dist_map)
        out_file = Path(target_batch_dir) / manifest_filename
        with open(out_file, 'w', encoding='utf-8') as f:
            f.write(uni_content)
        artifacts_written["universal_manifest"] = str(out_file)
        logger.info(f"\n✅ Wrote universal repository manifest to '{out_file}'")

    if args.output or args.in_place or args.output_dir:
        active_suffix_display = args.suffix if args.suffix is not None else ("" if args.output_dir else "_merged")
        if args.in_place:
            loc_desc = "in-place"
        elif args.output_dir:
            loc_desc = f"directory: '{args.output_dir}'" + (f", suffix: '{active_suffix_display}'" if active_suffix_display else "")
        else:
            loc_desc = f"suffix: '{active_suffix_display}'"

        logger.info(f"\n🚀 Writing per-notebook locked files ({loc_desc})...")
        written_files = []
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
                root_dir=repo_map.target_dir,
                output_dir=args.output_dir
            )
            written_files.append(str(written_path))
            logger.info(f"  • Updated '{written_path}'")
        artifacts_written["locked_notebooks"] = written_files
        logger.info("✅ Batch output complete.")

    if is_json:
        print(format_json_batch_report(summary, artifacts_written=artifacts_written if artifacts_written else None))

    return


def run_single_file_pipeline(
    args: argparse.Namespace, 
    frozen_env: Dict[str, str], 
    raw_full_freeze: List[str],
    pkg_dist_map: Dict[str, List[str]],
    precomputed_gpu_info: Optional[GpuInfo] = None
) -> None:
    """Executes single-notebook analysis or live IPython kernel history extraction."""
    in_live_ipython = is_running_in_ipython()
    is_json = getattr(args, "format", "text") == "json"

    target_single_file_dir = str(Path(args.notebook).parent) if (args.notebook and not os.path.isdir(args.notebook)) else "."
    single_file_local_modules = discover_local_repo_modules(target_single_file_dir)

    if args.notebook and not os.path.isdir(args.notebook):
        logger.info(f"🔍 [Path A] Analyzing saved notebook file '{args.notebook}' via AST...")
        logger.info(f"📌 Active Python Interpreter: {sys.executable}\n")
        
        ext_res = extract_from_file(args.notebook, strict=False)
        if not ext_res.success:
            logger.error(f"❌ Error: {ext_res.error_msg}")
            if is_json:
                bad_report = NotebookAnalysisReport(
                    notebook_path=str(args.notebook),
                    is_python=False,
                    lang_label=ext_res.lang_label,
                    parse_error=ext_res.error_msg
                )
                print(format_json_single_report(bad_report))
            if in_live_ipython:
                return
            sys.exit(1)
            
        imports, submodules, code_sources = ext_res.imports, ext_res.submodules, ext_res.code_sources
        guarded_imports, dyn_warnings = ext_res.guarded_imports, ext_res.dynamic_warnings
        writefile_imports = ext_res.writefile_imports
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
        scoped_flags=h_res.scoped_flags,
        magic_warnings=magic_warns,
        magic_notices=magic_notices
    )

    nb_report = build_single_notebook_report(
        single_res, frozen_env, pkg_dist_map, gpu_info, local_repo_modules=single_file_local_modules
    )

    if not is_json:
        if nb_report.warnings:
            logger.warning("⚠️ DIAGNOSTIC WARNINGS:")
            for warn in nb_report.warnings:
                logger.warning(f"  • {warn.detail}")
            logger.warning("")

        if nb_report.notices:
            for notice in nb_report.notices:
                logger.info(notice.format_console())
            logger.info("")

        if gpu_info:
            if gpu_info.has_gpu:
                logger.info(f"⚡ Active accelerator detected: {gpu_info.device_name}\n")
            elif gpu_info.probe_errors:
                err_msg = "; ".join(gpu_info.probe_errors)
                logger.warning(f"⚠️ Accelerator detection encountered errors: {err_msg}\n")
            elif gpu_info.frameworks:
                fw_list = ", ".join(gpu_info.frameworks)
                logger.warning(f"⚠️ Acceleration Framework ({fw_list}) imported, but NO active accelerator detected in host runtime.\n")

        if nb_report.promotions:
            for promo in nb_report.promotions:
                logger.info(promo.detail)
            logger.info("")

    artifacts_written: Optional[Dict[str, Any]] = None

    if args.output or args.in_place or args.output_dir:
        active_suffix_display = args.suffix if args.suffix is not None else ("" if args.output_dir else "_merged")
        if args.in_place:
            loc_desc = "in-place"
        elif args.output_dir:
            loc_desc = f"directory: '{args.output_dir}'" + (f", suffix: '{active_suffix_display}'" if active_suffix_display else "")
        else:
            loc_desc = f"suffix: '{active_suffix_display}'"

        logger.info(f"🚀 Writing updated notebook ({loc_desc})...")
        written_path = apply_output_to_notebook(
            single_res,
            frozen_env,
            pkg_dist_map,
            gpu_info,
            suffix=args.suffix,
            in_place=args.in_place,
            local_repo_modules=single_file_local_modules,
            root_dir=target_single_file_dir,
            output_dir=args.output_dir
        )
        artifacts_written = {"locked_notebook": str(written_path)}
        logger.info(f"✅ Updated '{written_path}'")
        if is_json:
            print(format_json_single_report(nb_report, artifacts_written=artifacts_written))
        if in_live_ipython:
            return
        return

    if is_json:
        print(format_json_single_report(nb_report))
        if in_live_ipython:
            return
        return

    full_freeze_lines = raw_full_freeze if args.full_freeze else None
    blueprint = generate_production_blueprint(
        nb_report.dependencies, 
        full_freeze_lines=full_freeze_lines, 
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
    get_notebook_local_modules.cache_clear()
    build_manifest_entries.cache_clear()

    parser = argparse.ArgumentParser(description="Generate environment lockfiles for Jupyter Notebooks.")
    parser.add_argument("notebook", nargs="?", help="Path to target .ipynb file or directory (when using --batch).")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output report format (default: 'text').")
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
    parser.add_argument("--output-dir", metavar="DIR", help="Directory where generated locked notebooks should be written.")
    parser.add_argument("--suffix", default=None, help="File suffix for merged notebook outputs (default: '_merged' alongside source, '' with --output-dir).")
    parser.add_argument("--in-place", action="store_true", help="Overwrite original notebooks in-place instead of creating companion files.")

    args, unknown = parser.parse_known_args()

    if is_running_in_ipython():
        sanitize_kernel_argv(args)

    if args.quiet:
        logger.setLevel(logging.ERROR)
    elif args.verbose:
        logger.setLevel(logging.DEBUG)

    target_batch_dir = args.batch or (args.notebook if args.notebook and os.path.isdir(args.notebook) else None)

    if (args.output or args.in_place or args.output_dir) and not target_batch_dir and not (args.notebook and os.path.isfile(args.notebook)):
        logger.error("❌ Error: --output, --output-dir, or --in-place requires a target notebook file path or --batch directory.")
        if is_running_in_ipython():
            return
        sys.exit(1)

    frozen_env, raw_full_freeze = get_installed_environment()
    pkg_dist_map = importlib.metadata.packages_distributions() if hasattr(importlib.metadata, "packages_distributions") else {}
    
    initial_imports: List[str] = []
    repo_map_pre: Optional[RepoEnvironmentMap] = None

    effective_suffix = args.suffix if args.suffix is not None else "_merged"
    skip_suffix = None if args.in_place else effective_suffix

    if target_batch_dir:
        repo_map_pre = walk_and_scan_directory(target_batch_dir, skip_suffix=skip_suffix)
        for imp in repo_map_pre.global_imports:
            if imp not in initial_imports:
                initial_imports.append(imp)
    elif args.notebook and os.path.isfile(args.notebook):
        ext_res = extract_from_file(args.notebook, strict=False)
        for imp in ext_res.imports:
            if imp not in initial_imports:
                initial_imports.append(imp)

    batch_hw_cache = inspect_gpu_environment(initial_imports)

    if target_batch_dir:
        run_batch_pipeline(target_batch_dir, args, frozen_env, pkg_dist_map, batch_hw_cache, precomputed_repo_map=repo_map_pre)
    else:
        run_single_file_pipeline(args, frozen_env, raw_full_freeze, pkg_dist_map, batch_hw_cache)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
PROJECT ENVIRONMENT-LOCK: NOTEBOOK SNAPSHOT TOOL (v25)

Headless Jupyter Notebook Dependency Scanner & Lockfile Generator.
Supports single-file processing and repository-wide batch execution (--batch).
"""

import ast
import json
import os
import re
import sys
import argparse
import subprocess
import importlib.metadata
from pathlib import Path
from datetime import datetime

# =====================================================================
# AST VISITOR & SOURCE PARSERS
# =====================================================================

class NotebookImportVisitor(ast.NodeVisitor):
    def __init__(self):
        self.imports = set()
        self.submodules = {}

    def visit_Import(self, node):
        for alias in node.names:
            base_pkg = alias.name.split('.')[0]
            self.imports.add(base_pkg)
            if '.' in alias.name:
                self.submodules.setdefault(base_pkg, set()).add(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            base_pkg = node.module.split('.')[0]
            self.imports.add(base_pkg)
            self.submodules.setdefault(base_pkg, set()).add(node.module)
        self.generic_visit(node)


def harvest_index_urls_from_sources(code_sources):
    """Scans code sources for --extra-index-url or -i flags."""
    index_urls = set()
    pattern = re.compile(r'(?:--extra-index-url|-i)\s+([^\s]+)')
    for source in code_sources:
        for line in source.splitlines():
            clean_line = line.strip()
            if clean_line.startswith('#'):
                continue
            matches = pattern.findall(clean_line)
            for url in matches:
                index_urls.add(url.strip("'\""))
    return index_urls


def extract_imports_from_sources(code_sources):
    """Extracts top-level imports and submodules via AST, stripping magics and shell commands."""
    visitor = NotebookImportVisitor()
    for source in code_sources:
        clean_source = "\n".join([
            line for line in source.splitlines() 
            if not line.strip().startswith('%') and not line.strip().startswith('!')
        ])
        try:
            tree = ast.parse(clean_source)
            visitor.visit(tree)
        except SyntaxError:
            continue

    return visitor.imports, visitor.submodules


def detect_notebook_language(nb_data, strict=False):
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
            return (ks_lang == "python"), ks_lang
        else:
            return False, f"conflict ({ks_lang}/{li_lang})"
    
    active_lang = ks_lang or li_lang
    if active_lang:
        return (active_lang == "python"), active_lang
        
    if strict:
        return False, "missing metadata"
    return True, "unspecified (assuming python)"


def extract_from_file(notebook_path, strict=False):
    if not os.path.exists(notebook_path):
        return False, set(), {}, [], f"File '{notebook_path}' not found.", "unknown"

    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb_data = json.load(f)
    except json.JSONDecodeError as e:
        return False, set(), {}, [], f"Invalid JSON structure ({e})", "corrupted"
    except Exception as e:
        return False, set(), {}, [], f"File read failure ({e})", "error"

    # 1. STRUCTURAL SCHEMA CHECK FIRST (Guarantees corrupt schema returns "corrupted")
    if not isinstance(nb_data, dict) or "cells" not in nb_data or not isinstance(nb_data.get("cells"), list):
        return False, set(), {}, [], "Unparseable notebook structure (Missing or invalid 'cells' array)", "corrupted"

    # 2. LANGUAGE CHECK SECOND
    is_py, lang_label = detect_notebook_language(nb_data, strict=strict)
    if not is_py:
        return False, set(), {}, [], f"Skipped non-Python notebook (Language: {lang_label})", lang_label

    cells = nb_data.get("cells", [])
    code_sources = ["".join(c.get("source", [])) for c in cells if c.get("cell_type") == "code"]
    imports, submodules = extract_imports_from_sources(code_sources)
    return True, imports, submodules, code_sources, None, lang_label

def extract_from_active_session():
    """Path B (Live Kernel): Reads IPython execution history."""
    import __main__
    code_sources = [src for src in getattr(__main__, 'In', []) if src and isinstance(src, str)]
    imports, submodules = extract_imports_from_sources(code_sources)
    return imports, submodules, code_sources


# =====================================================================
# ENVIRONMENT CORRELATION & HARDWARE INSPECTION
# =====================================================================

IMPORT_TO_PYPI_MAP = {
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "PIL": "Pillow",
    "yaml": "PyYAML",
    "bs4": "beautifulsoup4",
    "attr": "attrs",
    "serial": "pyserial"
}

STD_LIB = set(sys.stdlib_module_names) if hasattr(sys, 'stdlib_module_names') else {
    "os", "sys", "re", "json", "ast", "subprocess", "datetime", "math", "random", 
    "time", "pathlib", "typing", "collections", "itertools", "functools", "shutil"
}


def resolve_pypi_package_and_extras(imp, submodules_set, frozen_env, pkg_dist_map=None):
    """
    Resolves top-level import to PyPI package name using memoized metadata first.
    Promotes submodules to optional extras if declared in package metadata.
    """
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


def inspect_gpu_environment(imported_packages):
    """Per-framework GPU/accelerator inspection logic."""
    gpu_frameworks = {"torch", "tensorflow", "jax"}
    found_frameworks = gpu_frameworks.intersection(imported_packages)
    
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
                    "frameworks": list(found_frameworks)
                }
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return {
                    "has_gpu": True,
                    "type": "Apple Silicon MPS",
                    "active_framework": "PyTorch",
                    "device_name": "Apple Silicon GPU (Metal via PyTorch)",
                    "frameworks": list(found_frameworks)
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
                    "frameworks": list(found_frameworks)
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
                    "frameworks": list(found_frameworks)
                }
        except Exception:
            pass

    return {
        "has_gpu": False,
        "type": None,
        "active_framework": None,
        "device_name": None,
        "frameworks": list(found_frameworks)
    }


def resolve_opencv_variant(submodules=None):
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


def get_installed_environment():
    """Runs pip freeze to get precise version snapshots."""
    res = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    frozen = {}
    for line in res.stdout.splitlines():
        if "==" in line:
            pkg, ver = line.split("==", 1)
            frozen[pkg.lower()] = line.strip()
    return frozen, res.stdout.splitlines()


def process_package_requirements(pinned_list, harvested_urls):
    """Correlates pinned packages with harvested index URLs."""
    manifest_output = []
    local_tagged_info = []
    warnings = []
    
    if harvested_urls:
        for url in sorted(harvested_urls):
            manifest_output.append(f"--extra-index-url {url}")

    for item in pinned_list:
        manifest_output.append(item)
        if '+' in item:
            local_tagged_info.append((item, list(harvested_urls)))
            if not harvested_urls:
                warnings.append(item)
            
    return manifest_output, local_tagged_info, warnings


# =====================================================================
# BLUEPRINT & CELL METADATA GENERATOR
# =====================================================================

def generate_production_blueprint(manifest_lines, full_freeze_lines=None, local_tagged_info=None, gpu_info=None):
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


def create_managed_cells(blueprint):
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
# BATCH ORCHESTRATION ENGINE
# =====================================================================

class NotebookScanResult:
    def __init__(self, path, is_python, lang_label, parse_error=None, imports=None, submodules=None, code_sources=None):
        self.path = path
        self.is_python = is_python
        self.lang_label = lang_label
        self.parse_error = parse_error
        self.imports = imports or set()
        self.submodules = submodules or {}
        self.code_sources = code_sources or []
        self.harvested_urls = harvest_index_urls_from_sources(self.code_sources) if self.code_sources else set()


class RepoEnvironmentMap:
    def __init__(self, target_dir):
        self.target_dir = target_dir
        self.scan_results = []
        self.non_python_files = []
        self.parse_errors = []
        self.global_imports = set()
        self.package_to_notebooks = {}
        self.url_to_notebooks = {}
        self.promotions = []
        self.decisions = []

    def add_result(self, result):
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

        for url in result.harvested_urls:
            self.url_to_notebooks.setdefault(url, []).append(result.path)


def select_primary_index_url(url_to_notebooks):
    """
    Deterministically selects primary index URL:
    1. Majority notebook frequency rule
    2. Alphabetical filename tie-break
    3. Alphabetical URL string tie-break
    """
    if not url_to_notebooks:
        return None, None

    sorted_urls = sorted(url_to_notebooks.keys())
    
    def sorting_key(url):
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


def walk_and_scan_directory(target_dir):
    """Recursively scans directory for .ipynb files in strict batch mode."""
    repo_map = RepoEnvironmentMap(target_dir)
    target_path = Path(target_dir)

    for root, dirs, files in os.walk(target_path):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('venv', 'env', '__pycache__')]
        for file in sorted(files):
            if file.endswith('.ipynb'):
                full_path = Path(root) / file
                success, imports, submodules, code_sources, err, lang_label = extract_from_file(str(full_path), strict=True)
                
                parse_err = err if (not success and "Skipped non-Python notebook" not in (err or "")) else None                
                res = NotebookScanResult(
                    path=full_path,
                    is_python=success,
                    lang_label=lang_label,
                    parse_error=parse_err,
                    imports=imports,
                    submodules=submodules,
                    code_sources=code_sources
                )
                repo_map.add_result(res)

    return repo_map


def generate_batch_analysis_report(repo_map, frozen_env, pkg_dist_map, batch_hw_cache):
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
        lang_counts = {}
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

    # Dependency correlation
    matched_packages = set()
    missing_packages = {}
    promotions = []

    for res in repo_map.scan_results:
        for imp in sorted(res.imports):
            if imp in STD_LIB:
                continue
            submods = res.submodules.get(imp, set())
            pin_entry, notice = resolve_pypi_package_and_extras(imp, submods, frozen_env, pkg_dist_map=pkg_dist_map)
            
            if notice and notice not in promotions:
                promotions.append(notice)

            if pin_entry.startswith("#"):
                pypi_name = pin_entry.split()[1]
                missing_packages.setdefault(pypi_name, []).append(res.path.name)
            else:
                pkg_name = pin_entry.split("==")[0]
                matched_packages.add(pkg_name)

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

    if promotions:
        out.append("💡 DYNAMIC PROMOTIONS DETECTED:")
        for note in promotions:
            out.append(f"  • {note}")
        out.append("")

    # Hardware & Index Audit
    out.append("⚡ HARDWARE & INDEX URL AUDIT:")
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


def generate_universal_manifest(repo_map, frozen_env, pkg_dist_map):
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

    pinned_entries = set()
    for res in repo_map.scan_results:
        for imp in sorted(res.imports):
            if imp in STD_LIB:
                continue
            submods = res.submodules.get(imp, set())
            pin_entry, _ = resolve_pypi_package_and_extras(imp, submods, frozen_env, pkg_dist_map=pkg_dist_map)
            pinned_entries.add(pin_entry)

    for entry in sorted(pinned_entries):
        lines.append(entry)

    return "\n".join(lines)


def apply_output_to_notebook(scan_res, frozen_env, pkg_dist_map, batch_hw_cache, suffix="_merged", in_place=False):
    """Writes per-notebook locked file or replaces cells in-place."""
    pinned_manifest = []
    for imp in sorted(scan_res.imports):
        if imp in STD_LIB:
            continue
        submods = scan_res.submodules.get(imp, set())
        pin_entry, _ = resolve_pypi_package_and_extras(imp, submods, frozen_env, pkg_dist_map=pkg_dist_map)
        pinned_manifest.append(pin_entry)

    manifest_lines, local_tagged, _ = process_package_requirements(pinned_manifest, scan_res.harvested_urls)
    
    # Filter GPU info for this specific notebook
    gpu_info = None
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
        # Remove existing managed cells
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


# =====================================================================
# MAIN EXECUTION ENTRYPOINT
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate environment lockfiles for Jupyter Notebooks.")
    parser.add_argument("notebook", nargs="?", help="Path to target .ipynb file or directory (when using --batch).")
    parser.add_argument("--full-freeze", action="store_true", help="Append full environment pip freeze after targeted manifest.")
    
    # Batch Flags
    parser.add_argument("--batch", metavar="DIR", help="Run in batch mode across all notebooks in specified directory.")
    parser.add_argument("--analyze", action="store_true", help="Run batch analysis mode (default when --batch is provided).")
    parser.add_argument("--universal", action="store_true", help="Generate root requirements-all.txt universal manifest.")
    parser.add_argument("--output", action="store_true", help="Generate per-notebook merged lockfiles.")
    parser.add_argument("--suffix", default="_merged", help="File suffix for merged notebook outputs (default: '_merged').")
    parser.add_argument("--in-place", action="store_true", help="Overwrite original notebooks in-place instead of creating companion files.")

    args, unknown = parser.parse_known_args()

    # Memoized environment-level checks (Computed ONCE at entry)
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
            print("\n❌ Execution aborted: Resolve file/parse errors before running --universal or --output.")
            sys.exit(1)

        if args.universal:
            uni_content = generate_universal_manifest(repo_map, frozen_env, pkg_dist_map)
            out_file = Path(target_batch_dir) / "requirements-all.txt"
            with open(out_file, 'w', encoding='utf-8') as f:
                f.write(uni_content)
            print(f"\n✅ Wrote universal repository manifest to '{out_file}'")

        if args.output:
            print(f"\n🚀 Writing per-notebook locked files ({'in-place' if args.in_place else 'suffix: ' + args.suffix})...")
            for res in repo_map.scan_results:
                written_path = apply_output_to_notebook(
                    res, frozen_env, pkg_dist_map, batch_hw_cache, 
                    suffix=args.suffix, in_place=args.in_place
                )
                print(f"  • Updated '{written_path.name}'")
            print("✅ Batch output complete.")

        sys.exit(0)

    # --- SINGLE FILE / LIVE SESSION DISPATCH ---
    in_live_ipython = False
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            in_live_ipython = True
    except ImportError:
        pass

    if args.notebook and not os.path.isdir(args.notebook):
        print(f"🔍 [Path A] Analyzing saved notebook file '{args.notebook}' via AST...")
        print(f"📌 Active Python Interpreter: {sys.executable}\n")
        
        success, imports, submodules, code_sources, error_msg, _ = extract_from_file(args.notebook, strict=False)
        if not success:
            print(f"❌ Error: {error_msg}")
            sys.exit(1)
    elif in_live_ipython:
        print("🔍 [Path B] Analyzing live IPython session kernel history via AST...")
        imports, submodules, code_sources = extract_from_active_session()
    else:
        parser.print_help()
        sys.exit(1)

    harvested_urls = harvest_index_urls_from_sources(code_sources)
    gpu_info = inspect_gpu_environment(imports)
    
    pinned_manifest = []
    promotion_notices = []

    for imp in sorted(imports):
        if imp in STD_LIB:
            continue
        submods = submodules.get(imp, set())
        pin_entry, notice = resolve_pypi_package_and_extras(imp, submods, frozen_env, pkg_dist_map=pkg_dist_map)
        pinned_manifest.append(pin_entry)
        if notice:
            promotion_notices.append(notice)

    manifest_lines, local_tagged_info, warnings = process_package_requirements(pinned_manifest, harvested_urls)
    full_freeze_lines = raw_full_freeze if args.full_freeze else None

    # RESTORED TERMINAL WARNINGS: Local Tag Builds
    if warnings:
        print("⚠️ HARDWARE BUILD WARNINGS:")
        for pkg in warnings:
            print(f"  • Specific hardware build detected: `{pkg}`")
            print("    No matching download URL was found in code cells. If installation fails on target machines, ensure runtime matches or supply an --extra-index-url.\n")

    # RESTORED TERMINAL WARNINGS: GPU Acceleration Status
    if gpu_info:
        if gpu_info.get("has_gpu"):
            print(f"⚡ Active accelerator detected: {gpu_info['device_name']}\n")
        elif gpu_info.get("frameworks"):
            fw_list = ", ".join(gpu_info["frameworks"])
            print(f"⚠️ Acceleration Framework ({fw_list}) imported, but NO active accelerator detected in host runtime.\n")

    if promotion_notices:
        for note in promotion_notices:
            print(note)
        print()

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


if __name__ == "__main__":
    main()
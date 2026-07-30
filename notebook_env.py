#!/usr/bin/env python3
"""
PROJECT ENVIRONMENT-LOCK: NOTEBOOK SNAPSHOT TOOL (v22)

Headless Jupyter Notebook Dependency Scanner & Lockfile Generator.
Scans notebook imports via AST, dynamically correlates against active environment metadata,
promotes submodules to declared package extras, and generates self-contained setup blueprints.

Supports dual-path execution:
  - Path A (CLI / Disk): Parses saved .ipynb file on disk.
  - Path B (Live Session): Parses live IPython execution history (__main__.In).
"""

import ast
import json
import os
import re
import sys
import argparse
import subprocess
import importlib.metadata
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
    """
    Scans raw code strings to find --extra-index-url or -i flags.
    Ignores commented lines starting with '#' to prevent harvesting dead code.
    Returns a set of discovered index URL strings.
    """
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
    """
    Core AST parser: Takes a list of raw code strings (from disk cells OR live kernel history).
    Strips magics (%) and shell commands (!) prior to parsing.
    """
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


def extract_from_file(notebook_path):
    """
    Path A (CLI / Disk): Reads saved .ipynb file off disk.
    Returns: (success: bool, imports: set, submodules: dict, code_sources: list, error_msg: str)
    """
    if not os.path.exists(notebook_path):
        return False, set(), {}, [], f"File '{notebook_path}' not found."

    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb_data = json.load(f)
    except json.JSONDecodeError:
        return False, set(), {}, [], f"File '{notebook_path}' is not a valid Jupyter Notebook JSON format."
    except Exception as e:
        return False, set(), {}, [], f"Error reading notebook file: {e}"

    cells = nb_data.get("cells", [])
    code_sources = ["".join(c.get("source", [])) for c in cells if c.get("cell_type") == "code"]
    imports, submodules = extract_imports_from_sources(code_sources)
    return True, imports, submodules, code_sources, None


def extract_from_active_session():
    """
    Path B (Live Kernel): Reads IPython execution history (__main__.In).
    NOTE: Stale/deleted cells from earlier in the session remain until Kernel Restart.
    Returns: imports, submodules, code_sources
    """
    import __main__
    code_sources = [src for src in getattr(__main__, 'In', []) if src and isinstance(src, str)]
    imports, submodules = extract_imports_from_sources(code_sources)
    return imports, submodules, code_sources


# =====================================================================
# ENVIRONMENT CORRELATION & HARDWARE INSPECTION
# =====================================================================

# Fallback safety net map for packages NOT installed in current environment at snapshot time
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


def resolve_pypi_package_and_extras(imp, submodules_set, frozen_env):
    """
    Resolves top-level import to its canonical PyPI package name using live metadata first.
    Promotes submodules to optional extras if declared in package metadata (e.g. umap.plot -> umap-learn[plot]).
    Falls back to static map if the package is not installed in active environment.
    
    Returns: (pinned_manifest_entry, promotion_notice_str or None)
    """
    pypi_name = None
    try:
        if hasattr(importlib.metadata, "packages_distributions"):
            pkg_dist_map = importlib.metadata.packages_distributions()
            dists = pkg_dist_map.get(imp)
            if dists:
                pypi_name = dists[0]
    except Exception:
        pass

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
    """
    Per-framework GPU/accelerator inspection logic. Queries only the frameworks imported:
      - PyTorch: checks torch.cuda and torch.backends.mps
      - TensorFlow: checks tf.config.list_physical_devices('GPU')
      - JAX: checks jax.devices() for non-CPU platforms (GPU/TPU)
    Returns exact active_framework alongside device details.
    """
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
    """
    Determines the appropriate OpenCV package variant installed in the environment.
    """
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
    """
    Runs `pip freeze` to get precise version snapshots of the current runtime.
    """
    res = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    frozen = {}
    for line in res.stdout.splitlines():
        if "==" in line:
            pkg, ver = line.split("==", 1)
            frozen[pkg.lower()] = line.strip()
    return frozen, res.stdout.splitlines()


def process_package_requirements(pinned_list, harvested_urls):
    """
    Processes pinned packages, identifies local (+build) tags,
    and correlates them with harvested index URLs.
    Returns: (manifest_output, local_tagged_info, warnings)
    """
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
# BLUEPRINT GENERATOR
# =====================================================================

def generate_production_blueprint(manifest_lines, full_freeze_lines=None, local_tagged_info=None, gpu_info=None):
    """
    Assembles Cell 1 Markdown and Cell 2 Python code.
    Returns dict {"step1_markdown": str, "step2_code": str} for direct programmatic/test assertion.
    """
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


# =====================================================================
# MAIN EXECUTION ENTRYPOINT
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate environment lockfiles for Jupyter Notebooks.")
    parser.add_argument("notebook", nargs="?", help="Path to target .ipynb file (optional when running in live session).")
    parser.add_argument("--full-freeze", action="store_true", help="Append full environment pip freeze after targeted manifest.")
    
    args, unknown = parser.parse_known_args()

    in_live_ipython = False
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            in_live_ipython = True
    except ImportError:
        pass

    if args.notebook:
        print(f"🔍 [Path A] Analyzing saved notebook file '{args.notebook}' via AST...")
        print(f"📌 Active Python Interpreter: {sys.executable}")
        print("   (Verify this matches the environment/kernel used for your notebook)\n")
        
        success, imports, submodules, code_sources, error_msg = extract_from_file(args.notebook)
        if not success:
            print(f"❌ Error: {error_msg}")
            sys.exit(1)
    elif in_live_ipython:
        print("🔍 [Path B] Analyzing live IPython session kernel history via AST...")
        print("💡 Note: Always restart kernel & run all first to flush stale/deleted imports from session memory.\n")
        imports, submodules, code_sources = extract_from_active_session()
    else:
        parser.print_help()
        sys.exit(1)

    harvested_urls = harvest_index_urls_from_sources(code_sources)
    
    gpu_info = inspect_gpu_environment(imports)
    if gpu_info:
        if gpu_info["has_gpu"]:
            print(f"⚡ Active accelerator detected during snapshot: {gpu_info['device_name']}")
            print("   Captured device name for end-user Cell 1 Markdown.\n")
        else:
            frameworks_str = ", ".join(gpu_info["frameworks"])
            print(f"⚠️ Acceleration Framework ({frameworks_str}) imported, but NO active GPU/TPU accelerator was found!")
            print("   Your notebook imported an accelerator library, but hardware acceleration was not active during this run.")
            print("   If you intended to require a GPU/TPU, note that this test run executed on CPU.\n")

    frozen_env, raw_full_freeze = get_installed_environment()
    
    pinned_manifest = []
    promotion_notices = []

    for imp in sorted(imports):
        if imp in STD_LIB:
            continue
        
        submods = submodules.get(imp, set())
        pin_entry, notice = resolve_pypi_package_and_extras(imp, submods, frozen_env)
        
        pinned_manifest.append(pin_entry)
        if notice:
            promotion_notices.append(notice)

    if promotion_notices:
        for note in promotion_notices:
            print(note)
        print()

    manifest_lines, local_tagged_info, warnings = process_package_requirements(pinned_manifest, harvested_urls)
    
    if harvested_urls:
        print("ℹ️ Preserving download location(s) found in notebook cells:")
        for url in sorted(harvested_urls):
            print(f"   • {url}")
        print()

    for item in warnings:
        print(f"⚠️ Specific hardware build detected: '{item}'")
        print("   No download link was found in your notebook cells for this version.\n")
        print("   If students or reviewers run this notebook on a different platform,")
        print("   installation may fail unless you specify where to find this hardware build.\n")
        print("   To fix this, include the full download command in your setup cell like this:")
        print(f"   !pip install {item} --extra-index-url <YOUR_HARDWARE_INDEX_URL>\n")

    full_freeze_lines = raw_full_freeze if args.full_freeze else None

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
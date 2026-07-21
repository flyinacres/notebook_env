# =====================================================================
# PROJECT ENVIRONMENT-LOCK: NOTEBOOK SNAPSHOT TOOL (v15)
# Paste this into the LAST cell of your working notebook and run it.
# =====================================================================

import sys
import os
import ast
import urllib.request
import json
import datetime
import importlib.metadata
import importlib.util
from packaging.version import parse as parse_version

def clean_hardware_version(pkg_name, version_str):
    """
    Strips system-specific build tags (+cu121, +cpu) from frameworks like PyTorch
    for top-level targeted installs, allowing pip to select local machine drivers.
    """
    hardware_frameworks = {'torch', 'tensorflow', 'jax', 'cupy'}
    if pkg_name in hardware_frameworks and '+' in version_str:
        return version_str.split('+')[0]
    return version_str

def is_internet_available():
    """Fast network probe (0.5s timeout) to prevent silent timeouts on offline runtimes."""
    try:
        urllib.request.urlopen('https://pypi.org', timeout=0.5)
        return True
    except Exception:
        return False

def fetch_latest_pypi_version(pypi_name):
    """Fetches the current public release version from PyPI."""
    url = f"https://pypi.org/pypi/{pypi_name}/json"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Environment-Lock-Linter'})
        with urllib.request.urlopen(req, timeout=2) as response:
            data = json.loads(response.read().decode())
            return data.get('info', {}).get('version', None)
    except Exception:
        return None

def add_ast_parent_references(tree):
    """Attaches parent references to AST nodes to track try/except wrapper contexts."""
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child._parent = parent

def is_inside_try_block(node):
    """Identifies if an import lives inside a try/except block (optional check)."""
    curr = getattr(node, '_parent', None)
    while curr is not None:
        if isinstance(curr, ast.Try):
            return True
        curr = getattr(curr, '_parent', None)
    return False

def generate_production_blueprint(full_freeze=False):
    """
    Generates environment setup blocks for shareable notebooks.
    
    Parameters:
    -----------
    full_freeze : bool (default=False)
        - If False: Locks explicit top-level dependencies discovered via AST scan.
        - If True: Locks explicit top-level dependencies AND appends a complete 
          commented-out system snapshot at the bottom of the manifest.
    """
    print("?? Analyzing notebook imports and checking library versions...\n")
    
    errors = []
    outdated_packages = []
    imported_modules = {} 
    
    # 1. Inspect execution history stored in IPython memory
    try:
        import IPython
        ipython = IPython.get_ipython()
        history = ipython.user_ns.get('_ih', []) if ipython else []
    except Exception as e:
        print(f"? Could not access notebook cell history: {e}")
        return

    # 2. Parse code lines into AST nodes
    for cell_code in history:
        clean_lines = [line for line in cell_code.splitlines() if not line.strip().startswith(('%', '!'))]
        try:
            tree = ast.parse("\n".join(clean_lines))
            add_ast_parent_references(tree)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        parts = alias.name.split('.')
                        base_mod = parts[0]
                        
                        entry = imported_modules.setdefault(base_mod, {'subs': set(), 'is_optional': True})
                        if not is_inside_try_block(node):
                            entry['is_optional'] = False
                            
                        if len(parts) > 1:
                            entry['subs'].add(parts[1])

                elif isinstance(node, ast.ImportFrom) and node.module:
                    parts = node.module.split('.')
                    base_mod = parts[0]
                    
                    entry = imported_modules.setdefault(base_mod, {'subs': set(), 'is_optional': True})
                    if not is_inside_try_block(node):
                        entry['is_optional'] = False
                        
                    if len(parts) > 1:
                        entry['subs'].add(parts[1])
                    for alias in node.names:
                        entry['subs'].add(alias.name)
        except SyntaxError:
            continue

    # 3. Filter out standard library modules
    stdlib_modules = getattr(sys, "stdlib_module_names", set())
    internal_tools = {'sys', 'os', 'socket', 'ast', 'importlib', 'IPython', 'json', 'datetime', 'packaging', 'urllib'}
    
    filtered_modules = {
        mod: info for mod, info in imported_modules.items() 
        if mod not in stdlib_modules and mod not in internal_tools
    }

    if not filtered_modules:
        print("!"*80)
        print("??  No external library imports were found in this session.")
        print("?? How to fix: Run your notebook code cells first, then re-run this tool.")
        print("!"*80)
        return

    # 4. Map import names to installed PyPI packages
    try:
        pkg_dist_map = importlib.metadata.packages_distributions()
    except Exception as e:
        print(f"? Failed to read system package metadata: {e}")
        return

    detected_packages = {}

    for root_module, info in sorted(filtered_modules.items()):
        submodules = info['subs']
        is_optional = info['is_optional']

        try:
            spec = importlib.util.find_spec(root_module)
        except Exception:
            spec = None
            
        if spec is None:
            if is_optional:
                continue
            else:
                errors.append(f"? Missing Package: You imported '{root_module}', but it is not installed in this environment.")
                continue

        if root_module not in pkg_dist_map or not pkg_dist_map[root_module]:
            errors.append(f"? Unmapped Package: '{root_module}' is installed, but its installer metadata could not be found.")
            continue

        for pypi_name in pkg_dist_map[root_module]:
            try:
                version_str = importlib.metadata.version(pypi_name)
                
                # Targeted Extra Variant Match
                matched_target = pypi_name
                if submodules:
                    try:
                        metadata = importlib.metadata.metadata(pypi_name)
                        provides_extra = metadata.get_all('Provides-Extra', [])
                        if provides_extra:
                            for sub in submodules:
                                if sub in provides_extra:
                                    matched_target = f"{pypi_name}[{sub}]"
                                    break
                    except Exception:
                        pass
                
                detected_packages[matched_target] = version_str
                
            except importlib.metadata.PackageNotFoundError:
                errors.append(f"? Missing Version: Version details for '{pypi_name}' could not be read.")

    manifest_lines = []

    # Fast offline probe gatekeeper
    has_internet = is_internet_available()
    if not has_internet:
        print("?? Offline mode: Skipping online package freshness check.\n")

    # 5. Validate targeted versions and run optional freshness checks
    for pkg_target, ver in sorted(detected_packages.items()):
        base_pkg_name = pkg_target.split('[')[0]
        
        if "/" in ver or "\\" in ver:
            errors.append(f"? Local Path Warning: '{base_pkg_name}' uses a custom local file path ('{ver}').")
            continue
        try:
            parse_version(ver)
        except Exception:
            errors.append(f"? Non-standard Version: '{base_pkg_name}' uses an unrecognized version format ('{ver}').")
            continue

        cleaned_local_ver = clean_hardware_version(base_pkg_name, ver)
        
        if has_internet:
            latest_ver = fetch_latest_pypi_version(base_pkg_name)
            if latest_ver and parse_version(cleaned_local_ver) < parse_version(latest_ver):
                outdated_packages.append((base_pkg_name, cleaned_local_ver, latest_ver))

        manifest_lines.append(f"{pkg_target}=={cleaned_local_ver}")

    if errors:
        print("!"*80)
        print("?? CANNOT GENERATE SETUP BLOCKS")
        print("Please resolve the following issues before sharing this notebook:")
        print("!"*80 + "\n")
        for err in errors:
            print(err)
        print("\n" + "!"*80)
        return

    # 6. Optional Additive Full Freeze Generation
    full_freeze_lines = []
    if full_freeze:
        full_freeze_lines.append("\n# =====================================================================")
        full_freeze_lines.append("# FULL ENVIRONMENT SNAPSHOT (ALL INSTALLED PACKAGES)")
        full_freeze_lines.append("# Uncomment individual lines below if you need to recreate the entire container bit-for-bit.")
        full_freeze_lines.append("# =====================================================================")
        
        for dist in sorted(importlib.metadata.distributions(), key=lambda d: d.metadata['Name'].lower()):
            p_name = dist.metadata['Name']
            p_ver = dist.version
            
            # Soft-comment problematic or local versions in full snapshot without crashing script
            if "/" in p_ver or "\\" in p_ver:
                full_freeze_lines.append(f"# local-path: {p_name}=={p_ver}")
            else:
                try:
                    parse_version(p_ver)
                    full_freeze_lines.append(f"# {p_name}=={p_ver}")
                except Exception:
                    full_freeze_lines.append(f"# invalid-version: {p_name}=={p_ver}")

    # Print compact freshness summary if outdated packages exist
    if outdated_packages:
        print("-" * 80)
        print("??  NOTICE: SOME LIBRARIES IN THIS ENVIRONMENT ARE OUTDATED")
        print("The setup block was generated using your current versions, but updating them")
        print("before sharing can prevent issues caused by older background code.")
        print("-" * 80)
        print(f"{'PACKAGE':<25} {'INSTALLED':<15} {'LATEST ONLINE':<15}")
        print("-" * 55)
        for pkg, curr, online in outdated_packages:
            print(f"{pkg:<25} {curr:<15} {online:<15}")
        print("-" * 80 + "\n")

    timestamp = datetime.date.today().strftime("%Y-%m-%d")
    
    print("="*80)
    print("?? HOW TO USE THIS OUTPUT:")
    print("1. Create two empty cells at the VERY TOP of your notebook.")
    print("2. Change Cell 1 to 'Markdown' and paste STEP 1 into it.")
    print("3. Keep Cell 2 as 'Code' and paste STEP 2 into it.")
    print("="*80 + "\n")

    print("--- [ STEP 1: PASTE INTO CELL 1 (MARKDOWN) ] ---\n")
    print(f"""# ??? Environment Alignment & Setup

This notebook includes an explicit library configuration block designed to match the environment used during its creation.

### ?? Why is this setup cell here?
* **Reduces Compatibility Issues:** Locking core library versions helps prevent errors caused when underlying packages release breaking changes later on.
* **Single-Pass Installation:** Installs required dependencies together to allow the package manager to find compatible versions before code executes.""")

    print("\n" + "-"*80 + "\n")
    print("--- [ STEP 2: PASTE INTO CELL 2 (CODE) ] ---\n")
    
    payload_string = "\n".join(manifest_lines).strip()
    if full_freeze_lines:
        payload_string += "\n" + "\n".join(full_freeze_lines)

    print(f"""# =====================================================================
# VERIFIED ENVIRONMENT DEPENDENCIES ({timestamp})
# =====================================================================

# Write explicit library requirements to a local file
requirements_content = \"\"\"# Tested top-level packages for this notebook
# Note: If an installation fails in the future, check whether a sub-dependency 
# introduced a breaking change or update the specific version tag below.
{payload_string}
\"\"\"

with open("pinned_requirements.txt", "w") as f:
    f.write(requirements_content.strip())

# Run single-pass installation without forcing unwanted upgrades to pre-installed tools
import sys
print(f"-> Running on Python {{sys.version.split()[0]}}")
print("Syncing notebook dependencies...")

!pip install -r pinned_requirements.txt
print("\\n? Setup complete! Environment ready.")""")
    print("\n" + "="*80)

# Execution entry point
generate_production_blueprint(full_freeze=False)

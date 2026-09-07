#!/usr/bin/env python3
"""
Phase 5f: Hardware Mocking Test.
Dynamically generates a mock torch package and a notebook, runs notebook_env.py, 
and verifies the generated Markdown correctly identifies the mocked hardware.
"""
import sys
import json
import subprocess
import os
import shutil

def main() -> int:
    mode = os.environ.get("TEST_HW_MODE", "none")
    fixture_path = "tests/fixtures/temp_hw_fixture.ipynb"
    mock_dir = "tests/fixtures/mock_pkgs/torch"
    mock_init = os.path.join(mock_dir, "__init__.py")
    
    # 1. Dynamically generate the mock torch package
    os.makedirs(mock_dir, exist_ok=True)
    with open(mock_init, "w", encoding="utf-8") as f:
        f.write("""
import os
__version__ = "2.3.0+mock"
class _CUDA:
    @staticmethod
    def is_available(): return os.environ.get("MOCK_CUDA_AVAILABLE", "0") == "1"
    @staticmethod
    def get_device_name(device=None): return "NVIDIA A100-SXM4-40GB (Mock)"
class _MPS:
    @staticmethod
    def is_available(): return os.environ.get("MOCK_MPS_AVAILABLE", "0") == "1"
    @staticmethod
    def is_built(): return os.environ.get("MOCK_MPS_AVAILABLE", "0") == "1"
class _Backends:
    mps = _MPS()
cuda = _CUDA()
backends = _Backends()
""")

    # 2. Generate the dummy notebook
    fixture_content = {
        "cells": [
            {
                "cell_type": "code",
                "source": ["import torch"],
                "metadata": {}
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    with open(fixture_path, "w", encoding="utf-8") as f:
        json.dump(fixture_content, f)
        
    print(f"1. Testing hardware mode: {mode.upper()}...")
    try:
        # sys.executable inherits the PYTHONPATH set in run_suite.ps1
        result = subprocess.run(
            [sys.executable, "notebook_env.py", fixture_path, "--in-place"],
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        print(f"FAIL: notebook_env.py execution failed.\n{e.stderr}\n{e.stdout}")
        return 1
    finally:
        # Clean up the dynamically generated mock package so it doesn't pollute the workspace
        if os.path.exists("tests/fixtures/mock_pkgs"):
            shutil.rmtree("tests/fixtures/mock_pkgs")

    # 3. Verify the output
    with open(fixture_path, "r", encoding="utf-8") as f:
        out_nb = json.load(f)
        
    os.remove(fixture_path)
        
    if not out_nb.get("cells"):
        print("FAIL: No cells found in output notebook.")
        return 1
        
    md_cell = "".join(out_nb["cells"][0]["source"]).lower()
    
    if mode == "cuda":
        if "a100" not in md_cell and "cuda" not in md_cell:
            print(f"FAIL: CUDA hardware not detected in Markdown.\nGenerated Markdown:\n{md_cell}")
            return 1
        print("   PASS: CUDA hardware correctly detected and documented.")
        
    elif mode == "mps":
        if "mps" not in md_cell and "apple" not in md_cell and "metal" not in md_cell:
            print(f"FAIL: MPS (Apple Silicon) hardware not detected in Markdown.\nGenerated Markdown:\n{md_cell}")
            return 1
        print("   PASS: Apple Silicon (MPS) hardware correctly detected and documented.")
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
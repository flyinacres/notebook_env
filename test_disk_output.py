import json
import sys
import subprocess
import pytest
from pathlib import Path

import notebook_env as ne


# --- FIXTURES ---

@pytest.fixture
def mock_frozen_env(monkeypatch):
    """Provides deterministic environment pins for Tier 1 tests."""
    mock_env = {
        "pandas": "pandas==2.1.0",
        "numpy": "numpy==1.25.0",
        "requests": "requests==2.31.0"
    }
    monkeypatch.setattr(
        ne, 
        "get_installed_environment", 
        lambda: (mock_env, ["pandas==2.1.0", "numpy==1.25.0", "requests==2.31.0"])
    )
    return mock_env


@pytest.fixture
def sample_notebook_data():
    """Generates standard Jupyter Notebook JSON dict with custom metadata."""
    return {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "outputs": [],
                "source": ["import pandas as pd\n", "import numpy as np\n"]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


@pytest.fixture
def sample_notebook_file(tmp_path, sample_notebook_data):
    """Writes a sample notebook to disk in a temporary directory."""
    nb_path = tmp_path / "test_notebook.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(sample_notebook_data, f, indent=1)
    return nb_path


# =====================================================================
# TIER 1: IN-PROCESS DISK & CONTENT TESTS (Mocked Environment)
# =====================================================================

def test_apply_output_companion_file(sample_notebook_file, mock_frozen_env):
    """Verify companion file creation (_merged.ipynb) without altering original notebook."""
    scan_res = ne.NotebookScanResult(
        path=sample_notebook_file,
        is_python=True,
        lang_label="python",
        imports={"pandas", "numpy"},
        code_sources=["import pandas as pd\nimport numpy as np"]
    )

    out_path = ne.apply_output_to_notebook(scan_res, mock_frozen_env, {}, None, suffix="_merged", in_place=False)

    assert out_path.exists()
    assert out_path.name == "test_notebook_merged.ipynb"

    # Original untouched
    with open(sample_notebook_file, "r", encoding="utf-8") as f:
        orig_data = json.load(f)
    assert len(orig_data["cells"]) == 1

    # Companion contains 2 managed cells + 1 original cell
    with open(out_path, "r", encoding="utf-8") as f:
        merged_data = json.load(f)

    assert len(merged_data["cells"]) == 3
    assert merged_data["metadata"]["kernelspec"]["name"] == "python3"
    assert merged_data["cells"][0]["metadata"]["notebook_env"]["managed"] is True
    assert merged_data["cells"][1]["metadata"]["notebook_env"]["managed"] is True


def test_apply_output_companion_overwrite_existing(sample_notebook_file, mock_frozen_env):
    """Verify that --output overwrites an existing companion file cleanly."""
    scan_res = ne.NotebookScanResult(
        path=sample_notebook_file,
        is_python=True,
        lang_label="python",
        imports={"pandas"},
        code_sources=["import pandas as pd"]
    )

    # First run
    ne.apply_output_to_notebook(scan_res, mock_frozen_env, {}, None, suffix="_merged", in_place=False)

    # Second run (overwriting existing _merged.ipynb)
    out_path2 = ne.apply_output_to_notebook(scan_res, mock_frozen_env, {}, None, suffix="_merged", in_place=False)

    assert out_path2.exists()
    with open(out_path2, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data["cells"]) == 3

def test_apply_output_gpu_misattribution_prevented(sample_notebook_file, mock_frozen_env):
    """Verify that a TensorFlow notebook does not inherit PyTorch CUDA device attributions from batch cache."""
    scan_res = ne.NotebookScanResult(
        path=sample_notebook_file,
        is_python=True,
        lang_label="python",
        imports={"tensorflow"},
        code_sources=["import tensorflow as tf"]
    )

    # Batch HW cache populated by a PyTorch notebook on CUDA
    pytorch_batch_cache: ne.GpuInfo = {
        "has_gpu": True,
        "type": "NVIDIA CUDA",
        "active_framework": "PyTorch",
        "device_name": "NVIDIA GeForce RTX 4090 (via PyTorch)",
        "frameworks": ["torch", "tensorflow"]
    }

    out_path = ne.apply_output_to_notebook(
        scan_res, 
        mock_frozen_env, 
        {}, 
        pytorch_batch_cache, 
        in_place=True
    )

    with open(out_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    markdown_cell_source = "".join(data["cells"][0]["source"])
    
    # Assert PyTorch CUDA device name is NOT misattributed to the TensorFlow notebook
    assert "NVIDIA GeForce RTX 4090 (via PyTorch)" not in markdown_cell_source

def test_apply_output_inplace(sample_notebook_file, mock_frozen_env):
    """Verify in-place modification updates the target file directly."""
    scan_res = ne.NotebookScanResult(
        path=sample_notebook_file,
        is_python=True,
        lang_label="python",
        imports={"pandas"},
        code_sources=["import pandas as pd"]
    )

    out_path = ne.apply_output_to_notebook(scan_res, mock_frozen_env, {}, None, in_place=True)

    assert out_path == sample_notebook_file
    with open(sample_notebook_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data["cells"]) == 3
    assert data["cells"][0]["metadata"]["notebook_env"]["managed"] is True
    assert data["cells"][1]["metadata"]["notebook_env"]["managed"] is True


def test_inplace_idempotency_rerun(sample_notebook_file, mock_frozen_env):
    """Verify executing --in-place twice replaces managed cells without duplication, using real AST re-scan."""
    # First Pass
    success1, imports1, submodules1, sources1, _, _, guarded1, dyn1 = ne.extract_from_file(str(sample_notebook_file))
    scan_res1 = ne.NotebookScanResult(
        path=sample_notebook_file,
        is_python=success1,
        lang_label="python",
        imports=imports1,
        submodules=submodules1,
        guarded_imports=guarded1,
        dynamic_warnings=dyn1,
        code_sources=sources1
    )
    ne.apply_output_to_notebook(scan_res1, mock_frozen_env, {}, None, in_place=True)

    # Verify First Run Output
    with open(sample_notebook_file, "r", encoding="utf-8") as f:
        data_run1 = json.load(f)
    assert len(data_run1["cells"]) == 3

    # Second Pass: Perform genuine extract_from_file on the modified notebook
    success2, imports2, submodules2, sources2, _, _, guarded2, dyn2 = ne.extract_from_file(str(sample_notebook_file))
    scan_res2 = ne.NotebookScanResult(
        path=sample_notebook_file,
        is_python=success2,
        lang_label="python",
        imports=imports2,
        submodules=submodules2,
        guarded_imports=guarded2,
        dynamic_warnings=dyn2,
        code_sources=sources2
    )
    ne.apply_output_to_notebook(scan_res2, mock_frozen_env, {}, None, in_place=True)

    # Verify Second Run Output: Cell count must remain 3 (2 managed + 1 original user cell)
    with open(sample_notebook_file, "r", encoding="utf-8") as f:
        data_run2 = json.load(f)

    assert len(data_run2["cells"]) == 3
    assert data_run2["cells"][0]["metadata"]["notebook_env"]["managed"] is True
    assert data_run2["cells"][1]["metadata"]["notebook_env"]["managed"] is True


# =====================================================================
# TIER 2: SUBPROCESS CLI PLUMBING TESTS (Explicit UTF-8 Handles & Structural Checks)
# =====================================================================

def test_cli_single_file_inplace(sample_notebook_file):
    """CLI test for single-file --in-place execution without --output."""
    cmd = [sys.executable, "notebook_env.py", str(sample_notebook_file), "--in-place"]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")

    assert res.returncode == 0

    with open(sample_notebook_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data["cells"]) == 3
    assert data["cells"][0]["metadata"]["notebook_env"]["managed"] is True


def test_cli_single_file_output_companion(sample_notebook_file):
    """CLI test for single-file --output companion execution."""
    cmd = [sys.executable, "notebook_env.py", str(sample_notebook_file), "--output"]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")

    assert res.returncode == 0
    companion = sample_notebook_file.parent / "test_notebook_merged.ipynb"
    assert companion.exists()

    with open(companion, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data["cells"]) == 3


def test_cli_batch_inplace_alone(tmp_path, sample_notebook_data):
    """CLI test asserting --batch with --in-place alone (no --output) performs in-place writes."""
    nb1 = tmp_path / "nb1.ipynb"
    with open(nb1, "w", encoding="utf-8") as f:
        json.dump(sample_notebook_data, f)

    cmd = [sys.executable, "notebook_env.py", "--batch", str(tmp_path), "--in-place"]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")

    assert res.returncode == 0

    # Assert in-place modification occurred
    with open(nb1, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data["cells"]) == 3

    # Assert negative: No companion file nb1_merged.ipynb was created
    companion = tmp_path / "nb1_merged.ipynb"
    assert not companion.exists()


def test_cli_batch_output_and_inplace(tmp_path, sample_notebook_data):
    """CLI test asserting --in-place takes precedence when passed alongside --output."""
    nb1 = tmp_path / "nb1.ipynb"
    with open(nb1, "w", encoding="utf-8") as f:
        json.dump(sample_notebook_data, f)

    cmd = [sys.executable, "notebook_env.py", "--batch", str(tmp_path), "--output", "--in-place"]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")

    assert res.returncode == 0

    with open(nb1, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data["cells"]) == 3

    companion = tmp_path / "nb1_merged.ipynb"
    assert not companion.exists()


def test_cli_single_file_flags_without_notebook_errors():
    """CLI test asserting passing --in-place without a target notebook or --batch exits with error."""
    cmd = [sys.executable, "notebook_env.py", "--in-place"]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")

    assert res.returncode != 0
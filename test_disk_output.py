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
    monkeypatch.setattr(ne, "get_installed_environment", lambda: (mock_env, ["pandas==2.1.0", "numpy==1.25.0", "requests==2.31.0"]))
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
    """Verify executing --in-place twice replaces managed cells without duplication."""
    scan_res = ne.NotebookScanResult(
        path=sample_notebook_file,
        is_python=True,
        lang_label="python",
        imports={"pandas"},
        code_sources=["import pandas as pd"]
    )

    # First run
    ne.apply_output_to_notebook(scan_res, mock_frozen_env, {}, None, in_place=True)
    with open(sample_notebook_file, "r", encoding="utf-8") as f:
        data_run1 = json.load(f)
    assert len(data_run1["cells"]) == 3

    # Update scan_res with cells after first run
    code_sources = ["".join(c.get("source", [])) for c in data_run1["cells"] if c.get("cell_type") == "code"]
    scan_res_run2 = ne.NotebookScanResult(
        path=sample_notebook_file,
        is_python=True,
        lang_label="python",
        imports={"pandas"},
        code_sources=code_sources
    )

    # Second run
    ne.apply_output_to_notebook(scan_res_run2, mock_frozen_env, {}, None, in_place=True)
    with open(sample_notebook_file, "r", encoding="utf-8") as f:
        data_run2 = json.load(f)

    # Cell count must remain 3 (2 managed + 1 original user cell)
    assert len(data_run2["cells"]) == 3


# =====================================================================
# TIER 2: SUBPROCESS CLI PLUMBING TESTS (Explicit UTF-8 Subprocess Handles)
# =====================================================================

def test_cli_single_file_inplace(sample_notebook_file):
    """CLI test for single-file --in-place execution."""
    cmd = [sys.executable, "notebook_env.py", str(sample_notebook_file), "--in-place"]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")

    assert res.returncode == 0
    stdout = res.stdout or ""
    stderr = res.stderr or ""
    assert "Updated" in stdout or "Updated" in stderr

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


def test_cli_batch_output_and_inplace(tmp_path, sample_notebook_data):
    """CLI test for batch --batch --output --in-place execution."""
    nb1 = tmp_path / "nb1.ipynb"
    nb2 = tmp_path / "nb2.ipynb"
    with open(nb1, "w", encoding="utf-8") as f:
        json.dump(sample_notebook_data, f)
    with open(nb2, "w", encoding="utf-8") as f:
        json.dump(sample_notebook_data, f)

    cmd = [sys.executable, "notebook_env.py", "--batch", str(tmp_path), "--output", "--in-place"]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")

    assert res.returncode == 0

    for nb_path in (nb1, nb2):
        with open(nb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data["cells"]) == 3
        assert data["cells"][0]["metadata"]["notebook_env"]["managed"] is True
"""
Unit and Integration Tests for Batch Mode (v23).
Tests directory scanning, kernel language filtering, memoized metadata resolution,
primary index URL tie-break selection, and per-notebook output generation.
"""

import json
import sys
import types
import pytest
from pathlib import Path

import notebook_env as ne


@pytest.fixture
def mock_batch_env(monkeypatch):
    frozen_env = {
        "numpy": "numpy==1.26.4",
        "pandas": "pandas==2.2.1",
        "torch": "torch==2.3.1+cu121",
        "scikit-learn": "scikit-learn==1.4.2",
        "umap-learn": "umap-learn==0.5.5",
    }
    raw_freeze = list(frozen_env.values())
    pkg_dist_map = {
        "numpy": ["numpy"],
        "pandas": ["pandas"],
        "torch": ["torch"],
        "sklearn": ["scikit-learn"],
        "umap": ["umap-learn"]
    }
    monkeypatch.setattr(ne, "get_installed_environment", lambda: (frozen_env, raw_freeze))
    return frozen_env, pkg_dist_map


class TestLanguageKernelDetection:
    def test_python_notebook_detected(self):
        nb = {"metadata": {"kernelspec": {"language": "python"}}}
        is_py, label = ne.detect_notebook_language(nb)
        assert is_py is True
        assert label == "python"

    def test_r_notebook_skipped(self):
        nb = {"metadata": {"kernelspec": {"language": "R"}}}
        is_py, label = ne.detect_notebook_language(nb)
        assert is_py is False
        assert label == "r"

    def test_conflicting_language_metadata(self):
        nb = {"metadata": {"kernelspec": {"language": "python"}, "language_info": {"name": "julia"}}}
        is_py, label = ne.detect_notebook_language(nb)
        assert is_py is False
        assert "conflict" in label


class TestPrimaryIndexSelection:
    def test_majority_rule_selection(self):
        url_map = {
            "https://index.a.com": [Path("01.ipynb"), Path("02.ipynb"), Path("03.ipynb")],
            "https://index.b.com": [Path("04.ipynb")]
        }
        best_url, reason = ne.select_primary_index_url(url_map)
        assert best_url == "https://index.a.com"
        assert "Majority rule" in reason

    def test_alphabetical_filename_tie_break(self):
        url_map = {
            "https://index.b.com": [Path("02_file.ipynb")],
            "https://index.a.com": [Path("01_file.ipynb")]
        }
        best_url, reason = ne.select_primary_index_url(url_map)
        assert best_url == "https://index.a.com"

    def test_alphabetical_url_tie_break(self):
        url_map = {
            "https://z_index.com": [Path("01_same.ipynb")],
            "https://a_index.com": [Path("01_same.ipynb")]
        }
        best_url, reason = ne.select_primary_index_url(url_map)
        assert best_url == "https://a_index.com"


class TestBatchOrchestration:
    def test_batch_scan_and_universal_generation(self, tmp_path, mock_batch_env):
        frozen_env, pkg_dist_map = mock_batch_env
        
        # Create synthetic notebook 1
        nb1 = {
            "metadata": {
                "kernelspec": {"language": "python"}
            },
            "cells": [{"cell_type": "code", "source": ["import numpy as np\n!pip install -i https://index.foo.com pkg"]}]
        }
        nb1_path = tmp_path / "01_test.ipynb"
        nb1_path.write_text(json.dumps(nb1))

        # Create synthetic R notebook
        nb_r = {"metadata": {"kernelspec": {"language": "R"}}, "cells": []}
        (tmp_path / "02_r.ipynb").write_text(json.dumps(nb_r))

        repo_map = ne.walk_and_scan_directory(str(tmp_path))
        report, is_clean = ne.generate_batch_analysis_report(repo_map, frozen_env, pkg_dist_map, None)

        assert is_clean is True
        assert "Python (.ipynb): 1 files analyzed" in report
        assert "Non-Python skipped: 1 files" in report

        uni_manifest = ne.generate_universal_manifest(repo_map, frozen_env, pkg_dist_map)
        assert "--extra-index-url https://index.foo.com" in uni_manifest
        assert "numpy==1.26.4" in uni_manifest

    def test_per_notebook_output_generation(self, tmp_path, mock_batch_env):
        frozen_env, pkg_dist_map = mock_batch_env
        
        nb = {"cells": [{"cell_type": "code", "source": ["import pandas as pd\n"]}]}
        nb_path = tmp_path / "analysis.ipynb"
        nb_path.write_text(json.dumps(nb))

        res = ne.NotebookScanResult(path=nb_path, is_python=True, lang_label="python", imports={"pandas"}, code_sources=["import pandas as pd"])
        written_path = ne.apply_output_to_notebook(res, frozen_env, pkg_dist_map, None, suffix="_merged")

        assert written_path.exists()
        assert written_path.name == "analysis_merged.ipynb"

        out_nb = json.loads(written_path.read_text())
        first_cell = out_nb["cells"][0]
        assert first_cell["metadata"]["notebook_env"]["managed"] is True
        assert first_cell["metadata"]["notebook_env"]["role"] == "setup_markdown"

    def test_in_place_cell_replacement(self, tmp_path, mock_batch_env):
        frozen_env, pkg_dist_map = mock_batch_env
        
        # Notebook with pre-existing managed cell
        nb = {
            "cells": [
                {
                    "cell_type": "markdown", 
                    "metadata": {"notebook_env": {"managed": True, "role": "setup_markdown"}},
                    "source": ["OLD CONTENT"]
                },
                {"cell_type": "code", "source": ["import pandas as pd\n"]}
            ]
        }
        nb_path = tmp_path / "inplace_test.ipynb"
        nb_path.write_text(json.dumps(nb))

        res = ne.NotebookScanResult(path=nb_path, is_python=True, lang_label="python", imports={"pandas"}, code_sources=["import pandas as pd"])
        written_path = ne.apply_output_to_notebook(res, frozen_env, pkg_dist_map, None, in_place=True)

        out_nb = json.loads(written_path.read_text())
        # Managed cells replaced at indices 0 and 1, code cell preserved
        assert len(out_nb["cells"]) == 3
        assert "OLD CONTENT" not in out_nb["cells"][0]["source"][0]
        assert out_nb["cells"][0]["metadata"]["notebook_env"]["managed"] is True
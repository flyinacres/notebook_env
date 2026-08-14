"""
Unit and Integration Tests for Batch Mode (v25).
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

    def test_strict_mode_accepts_missing_metadata(self, tmp_path):
        nb_no_meta = {
            "cell_type": "code",
            "metadata": {},
            "cells": [{"cell_type": "code", "source": ["import math\n"]}]
        }
        nb_path = tmp_path / "no_metadata.ipynb"
        nb_path.write_text(json.dumps(nb_no_meta), encoding="utf-8")

        success, imports, submodules, code_sources, err, lang_label, guarded, dyn_warns = (
            ne.extract_from_file(str(nb_path), strict=True)
        )

        assert success is True
        assert "math" in imports
        assert err is None
        assert "unspecified" in lang_label or lang_label == ne.StatusLabel.PYTHON


class TestPrimaryIndexSelection:
    def test_majority_rule_selection(self):
        url_map = {
            "https://index.a.com": [Path("01.ipynb"), Path("02.ipynb"), Path("03.ipynb")],
            "https://index.b.com": [Path("04.ipynb")]
        }
        best_url, reason = ne.select_primary_index_url(url_map)
        assert best_url == "https://index.a.com"

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
        
        nb1 = {
            "metadata": {
                "kernelspec": {"language": "python"}
            },
            "cells": [{"cell_type": "code", "source": ["import numpy as np\n!pip install -i https://index.foo.com pkg"]}]
        }
        nb1_path = tmp_path / "01_test.ipynb"
        nb1_path.write_text(json.dumps(nb1))

        nb_r = {"metadata": {"kernelspec": {"language": "R"}}, "cells": []}
        (tmp_path / "02_r.ipynb").write_text(json.dumps(nb_r))

        repo_map = ne.walk_and_scan_directory(str(tmp_path))
        report, is_clean = ne.generate_batch_analysis_report(repo_map, frozen_env, pkg_dist_map, None)

        assert is_clean is True
        assert len(repo_map.scan_results) == 1
        assert len(repo_map.non_python_files) == 1

        uni_manifest = ne.generate_universal_manifest(repo_map, frozen_env, pkg_dist_map)
        assert "numpy==1.26.4" in uni_manifest
        assert "https://index.foo.com" in uni_manifest

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
        assert len(out_nb["cells"]) == 3
        assert "OLD CONTENT" not in out_nb["cells"][0]["source"][0]
        assert out_nb["cells"][0]["metadata"]["notebook_env"]["managed"] is True


    def test_batch_walk_populates_cell_magic_fields(self, tmp_path, mock_batch_env):
        frozen_env, pkg_dist_map = mock_batch_env
        
        nb = {
            "metadata": {"kernelspec": {"language": "python"}},
            "cells": [{
                "cell_type": "code", 
                "source": [
                    "%pip install seaborn\n",
                    "!pip install -i https://index.foo.com custom_pkg\n"
                ]
            }]
        }
        nb_path = tmp_path / "magic_test.ipynb"
        nb_path.write_text(json.dumps(nb))

        repo_map = ne.walk_and_scan_directory(str(tmp_path))
        assert len(repo_map.scan_results) == 1
        
        res = repo_map.scan_results[0]
        assert "seaborn" in res.harvested_pkgs
        assert "custom_pkg" in res.harvested_pkgs
        assert "https://index.foo.com" in res.base_index_urls
        assert "https://index.foo.com" in res.harvested_urls


    def test_batch_report_surfaces_unimported_magic_packages(self, tmp_path, mock_batch_env):
        frozen_env, pkg_dist_map = mock_batch_env
        
        nb = {
            "metadata": {"kernelspec": {"language": "python"}},
            "cells": [{"cell_type": "code", "source": ["!pip install gdown\n"]}]
        }
        nb_path = tmp_path / "unimported_magic.ipynb"
        nb_path.write_text(json.dumps(nb))

        repo_map = ne.walk_and_scan_directory(str(tmp_path))
        report, is_clean = ne.generate_batch_analysis_report(repo_map, frozen_env, pkg_dist_map, None)

        assert "gdown" in report


    def test_universal_manifest_includes_magic_packages(self, tmp_path, mock_batch_env):
        frozen_env, pkg_dist_map = mock_batch_env
        
        nb = {
            "metadata": {"kernelspec": {"language": "python"}},
            "cells": [{"cell_type": "code", "source": ["!pip install gdown\n"]}]
        }
        nb_path = tmp_path / "magic_manifest.ipynb"
        nb_path.write_text(json.dumps(nb))

        repo_map = ne.walk_and_scan_directory(str(tmp_path))
        uni_manifest = ne.generate_universal_manifest(repo_map, frozen_env, pkg_dist_map)

        assert "gdown" in uni_manifest


    def test_strict_mode_accepts_missing_metadata(self, tmp_path):
        nb_no_meta = {
            "cell_type": "code",
            "metadata": {},
            "cells": [{"cell_type": "code", "source": ["import math\n"]}]
        }
        nb_path = tmp_path / "no_metadata.ipynb"
        nb_path.write_text(json.dumps(nb_no_meta), encoding="utf-8")

        success, imports, submodules, code_sources, err, lang_label, guarded, dyn_warns = (
            ne.extract_from_file(str(nb_path), strict=True)
        )

        assert success is True
        assert "math" in imports
        assert err is None
        assert "unspecified" in lang_label or lang_label == ne.StatusLabel.PYTHON

    def test_batch_report_handles_local_tagged_builds(self, tmp_path, mock_batch_env):
        frozen_env, pkg_dist_map = mock_batch_env
        
        nb = {
            "metadata": {"kernelspec": {"language": "python"}},
            "cells": [{"cell_type": "code", "source": ["import torch\n"]}]
        }
        nb_path = tmp_path / "tagged_build.ipynb"
        nb_path.write_text(json.dumps(nb))

        repo_map = ne.walk_and_scan_directory(str(tmp_path))
        report, is_clean = ne.generate_batch_analysis_report(repo_map, frozen_env, pkg_dist_map, None)

        assert is_clean is True
        assert "torch" in report

    def test_platform_pseudo_modules_not_flagged_as_missing(self, tmp_path, mock_batch_env):
        frozen_env, pkg_dist_map = mock_batch_env

        nb = {
            "metadata": {"kernelspec": {"language": "python"}},
            "cells": [{"cell_type": "code", "source": ["import dbutils\nimport kaggle_secrets\n"]}]
        }
        nb_path = tmp_path / "pseudo_test.ipynb"
        nb_path.write_text(json.dumps(nb), encoding="utf-8")

        repo_map = ne.walk_and_scan_directory(str(tmp_path))
        report, is_clean = ne.generate_batch_analysis_report(repo_map, frozen_env, pkg_dist_map, None)

        assert "Packages missing from current environment: 0" in report

    def test_local_repo_modules_not_flagged_as_missing_pypi_packages(self, tmp_path, mock_batch_env):
        frozen_env, pkg_dist_map = mock_batch_env

        (tmp_path / "cookbook.py").write_text("# local helper file", encoding="utf-8")

        nb = {
            "metadata": {"kernelspec": {"language": "python"}},
            "cells": [{"cell_type": "code", "source": ["import cookbook\n"]}]
        }
        nb_path = tmp_path / "local_import_test.ipynb"
        nb_path.write_text(json.dumps(nb), encoding="utf-8")

        repo_map = ne.walk_and_scan_directory(str(tmp_path))
        report, is_clean = ne.generate_batch_analysis_report(repo_map, frozen_env, pkg_dist_map, None)

        assert "Packages missing from current environment: 0" in report

def test_batch_report_surfaces_hardware_tag_warnings(tmp_path):
    """Verify generate_batch_analysis_report flags local tag builds missing download index URLs."""
    nb_path = tmp_path / "test_hw_tag.ipynb"
    nb_data = {
        "cells": [{"cell_type": "code", "execution_count": 1, "metadata": {}, "outputs": [], "source": ["import torch"]}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5
    }
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb_data, f)

    mock_env = {"torch": "torch==2.1.0+cu121"}
    
    scan_res = ne.NotebookScanResult(
        path=nb_path,
        is_python=True,
        lang_label="python",
        imports={"torch"},
        code_sources=["import torch"]
    )
    
    repo_map = ne.RepoEnvironmentMap(str(tmp_path))
    repo_map.add_result(scan_res)
    
    report_text, _ = ne.generate_batch_analysis_report(repo_map, mock_env, {}, None)
    
    assert "Custom Build Tag Warnings:" in report_text

def test_batch_mode_scopes_local_modules_to_notebook_subdirectory(tmp_path):
    """Verify batch analysis recognizes local modules in subdirectories relative to the notebook."""
    # Build nested layout: repo_root/databricks/notebook.ipynb and repo_root/databricks/cookbook/__init__.py
    sub_dir = tmp_path / "databricks"
    cookbook_dir = sub_dir / "cookbook"
    cookbook_dir.mkdir(parents=True)
    (cookbook_dir / "__init__.py").write_text("# local package", encoding="utf-8")

    nb_path = sub_dir / "05_tool_calling_agent.ipynb"
    nb_data = {
        "cells": [{"cell_type": "code", "execution_count": 1, "metadata": {}, "outputs": [], "source": ["import cookbook"]}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5
    }
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb_data, f)

    scan_res = ne.NotebookScanResult(
        path=nb_path,
        is_python=True,
        lang_label="python",
        imports={"cookbook"},
        code_sources=["import cookbook"]
    )

    repo_map = ne.RepoEnvironmentMap(str(tmp_path))
    repo_map.add_result(scan_res)

    summary = ne.analyze_batch_repository(repo_map, {}, {}, None)

    # Assert 'cookbook' is recognized as a local repo module and NOT flagged as missing PyPI package
    assert "cookbook" not in summary.missing_packages
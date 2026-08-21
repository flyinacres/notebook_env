"""
Failure Modes and Edge Case Tests for Batch Mode (v26+).

Tests handling of unparseable JSON, bad notebook schemas, non-Python kernel filtering,
hidden/venv directory skipping, error-gated execution halts, multi-URL aggregation,
and low-level driver silence exception propagation.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

import notebook_env as ne
from notebook_env import StatusLabel


@pytest.fixture
def mock_batch_env(
    monkeypatch: pytest.MonkeyPatch
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """Provides a controlled mock environment for batch mode tests."""
    frozen_env: Dict[str, str] = {
        "numpy": "numpy==1.26.4",
        "pandas": "pandas==2.2.1",
        "torch": "torch==2.3.1+cu121",
    }
    raw_freeze: List[str] = list(frozen_env.values())
    pkg_dist_map: Dict[str, List[str]] = {
        "numpy": ["numpy"],
        "pandas": ["pandas"],
        "torch": ["torch"],
    }
    monkeypatch.setattr(ne, "get_installed_environment", lambda: (frozen_env, raw_freeze))
    return frozen_env, pkg_dist_map


class TestBatchFailureModes:
    """Tests directory scanning and batch orchestration failure modes."""

    def test_corrupted_json_blocks_execution(
        self, 
        tmp_path: Path, 
        mock_batch_env: Tuple[Dict[str, str], Dict[str, List[str]]]
    ) -> None:
        """A corrupted JSON file must populate parse_errors and halt batch execution."""
        frozen_env, pkg_dist_map = mock_batch_env

        valid_nb = {
            "metadata": {"kernelspec": {"language": "python"}},
            "cells": [{"cell_type": "code", "source": ["import numpy as np\n"]}]
        }
        (tmp_path / "01_valid.ipynb").write_text(json.dumps(valid_nb), encoding="utf-8")

        (tmp_path / "02_corrupted.ipynb").write_text("{ unquoted_json: True ", encoding="utf-8")

        repo_map = ne.walk_and_scan_directory(str(tmp_path))
        report, is_clean = ne.generate_batch_analysis_report(repo_map, frozen_env, pkg_dist_map, None)

        assert is_clean is False
        assert len(repo_map.parse_errors) == 1
        assert "02_corrupted.ipynb" in str(repo_map.parse_errors[0].path)

    def test_missing_cells_array_handled_as_corrupted(
        self, 
        tmp_path: Path, 
        mock_batch_env: Tuple[Dict[str, str], Dict[str, List[str]]]
    ) -> None:
        """JSON file lacking a 'cells' array should be logged as corrupted."""
        bad_schema = {"metadata": {"kernelspec": {"language": "python"}}}
        (tmp_path / "no_cells.ipynb").write_text(json.dumps(bad_schema), encoding="utf-8")

        repo_map = ne.walk_and_scan_directory(str(tmp_path))

        assert len(repo_map.parse_errors) == 1
        assert repo_map.parse_errors[0].lang_label in (StatusLabel.CORRUPTED, StatusLabel.ERROR)
        
        err_msg = repo_map.parse_errors[0].parse_error.lower()
        assert "cells" in err_msg or "unparseable" in err_msg

    def test_hidden_and_checkpoint_dirs_ignored(
        self, 
        tmp_path: Path, 
        mock_batch_env: Tuple[Dict[str, str], Dict[str, List[str]]]
    ) -> None:
        """Files inside .ipynb_checkpoints, .git, or venv should never be scanned."""
        frozen_env, pkg_dist_map = mock_batch_env

        root_nb = {
            "metadata": {"kernelspec": {"language": "python"}},
            "cells": [{"cell_type": "code", "source": ["import math\n"]}]
        }
        (tmp_path / "root.ipynb").write_text(json.dumps(root_nb), encoding="utf-8")

        checkpoint_dir = tmp_path / ".ipynb_checkpoints"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "root-checkpoint.ipynb").write_text("{ corrupted checkpoint json ", encoding="utf-8")

        venv_dir = tmp_path / "venv" / "lib"
        venv_dir.mkdir(parents=True)
        (venv_dir / "ignored.ipynb").write_text("{ corrupted venv json ", encoding="utf-8")

        repo_map = ne.walk_and_scan_directory(str(tmp_path))

        assert len(repo_map.parse_errors) == 0
        assert len(repo_map.scan_results) == 1
        assert repo_map.scan_results[0].path.name == "root.ipynb"

    def test_non_python_kernels_skipped_without_error(
        self, 
        tmp_path: Path, 
        mock_batch_env: Tuple[Dict[str, str], Dict[str, List[str]]]
    ) -> None:
        """R, Julia, and conflicting kernel metadata files must be skipped gracefully."""
        frozen_env, pkg_dist_map = mock_batch_env

        r_nb = {"metadata": {"kernelspec": {"language": "r"}}, "cells": []}
        julia_nb = {"metadata": {"kernelspec": {"language": "julia"}}, "cells": []}
        conflict_nb = {
            "metadata": {
                "kernelspec": {"language": "python"},
                "language_info": {"name": "r"}
            },
            "cells": []
        }

        (tmp_path / "script.r.ipynb").write_text(json.dumps(r_nb), encoding="utf-8")
        (tmp_path / "script.jl.ipynb").write_text(json.dumps(julia_nb), encoding="utf-8")
        (tmp_path / "conflict.ipynb").write_text(json.dumps(conflict_nb), encoding="utf-8")

        repo_map = ne.walk_and_scan_directory(str(tmp_path))
        report, is_clean = ne.generate_batch_analysis_report(repo_map, frozen_env, pkg_dist_map, None)

        assert is_clean is True
        assert len(repo_map.scan_results) == 0
        assert len(repo_map.non_python_files) == 3

    def test_cli_batch_aborts_on_parse_errors(
        self, 
        tmp_path: Path, 
        monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Calling --batch --universal on a directory with parse errors must exit with code 1."""
        (tmp_path / "broken.ipynb").write_text("{ invalid json ", encoding="utf-8")
        monkeypatch.setattr(sys, "argv", ["notebook_env.py", "--batch", str(tmp_path), "--universal"])

        with pytest.raises(SystemExit) as excinfo:
            ne.main()

        assert excinfo.value.code == 1

    def test_multiple_extra_index_urls_aggregated(
        self, 
        tmp_path: Path, 
        mock_batch_env: Tuple[Dict[str, str], Dict[str, List[str]]]
    ) -> None:
        """Multiple distinct index URLs across different notebooks must all appear in requirements-all.txt."""
        frozen_env, pkg_dist_map = mock_batch_env

        nb1 = {
            "metadata": {"kernelspec": {"language": "python"}},
            "cells": [{"cell_type": "code", "source": ["!pip install --extra-index-url https://index.a.com pkg1\n"]}]
        }
        nb2 = {
            "metadata": {"kernelspec": {"language": "python"}},
            "cells": [{"cell_type": "code", "source": ["!pip install --extra-index-url https://index.b.com pkg2\n"]}]
        }

        (tmp_path / "01.ipynb").write_text(json.dumps(nb1), encoding="utf-8")
        (tmp_path / "02.ipynb").write_text(json.dumps(nb2), encoding="utf-8")

        repo_map = ne.walk_and_scan_directory(str(tmp_path))
        manifest = ne.generate_universal_manifest(repo_map, frozen_env, pkg_dist_map)

        assert "--extra-index-url https://index.a.com" in manifest
        assert "--extra-index-url https://index.b.com" in manifest

    def test_batch_magic_requirements_file_warning_aggregated(self, tmp_path, mock_batch_env):
        frozen_env, pkg_dist_map = mock_batch_env
        nb = {
            "metadata": {"kernelspec": {"language": "python"}},
            "cells": [{"cell_type": "code", "source": ["!pip install -r requirements.txt\n"]}]
        }
        (tmp_path / "req_warn.ipynb").write_text(json.dumps(nb))

        repo_map = ne.walk_and_scan_directory(str(tmp_path))
        report, is_clean = ne.generate_batch_analysis_report(repo_map, frozen_env, pkg_dist_map, None)

        assert is_clean is True
        assert "requirements file" in report.lower()

    def test_batch_magic_conda_notice_aggregated(self, tmp_path, mock_batch_env):
        frozen_env, pkg_dist_map = mock_batch_env
        nb = {
            "metadata": {"kernelspec": {"language": "python"}},
            "cells": [{"cell_type": "code", "source": ["%conda install lightgbm\n"]}]
        }
        (tmp_path / "conda_notice.ipynb").write_text(json.dumps(nb))

        repo_map = ne.walk_and_scan_directory(str(tmp_path))
        report, is_clean = ne.generate_batch_analysis_report(repo_map, frozen_env, pkg_dist_map, None)

        assert is_clean is True
        assert "conda" in report.lower()

    def test_silence_fd2_stderr_does_not_crash_on_internal_exception(self):
        """
        Ensure silence_fd2_stderr allows exceptions inside the with block
        to propagate cleanly without raising 'RuntimeError: generator didn't stop after throw()'.
        """
        with pytest.raises(ValueError, match="Probe test error"):
            with ne.silence_fd2_stderr():
                raise ValueError("Probe test error")
"""
Failure Modes and Edge Case Tests for Batch Mode (v25).
Tests unparseable JSON, bad schema, non-Python kernel filtering, hidden dir skipping,
and error-gated execution halts.
"""

import json
import sys
import pytest
from pathlib import Path

import notebook_env as ne


@pytest.fixture
def mock_batch_env(monkeypatch):
    frozen_env = {
        "numpy": "numpy==1.26.4",
        "pandas": "pandas==2.2.1",
        "torch": "torch==2.3.1+cu121",
    }
    raw_freeze = list(frozen_env.values())
    pkg_dist_map = {
        "numpy": ["numpy"],
        "pandas": ["pandas"],
        "torch": ["torch"],
    }
    monkeypatch.setattr(ne, "get_installed_environment", lambda: (frozen_env, raw_freeze))
    return frozen_env, pkg_dist_map


class TestBatchFailureModes:

    def test_corrupted_json_blocks_execution(self, tmp_path, mock_batch_env, monkeypatch, capsys):
        """A corrupted JSON file must populate parse_errors and halt --universal or --output execution."""
        frozen_env, pkg_dist_map = mock_batch_env

        # 1 Valid notebook
        valid_nb = {
            "metadata": {"kernelspec": {"language": "python"}},
            "cells": [{"cell_type": "code", "source": ["import numpy as np\n"]}]
        }
        (tmp_path / "01_valid.ipynb").write_text(json.dumps(valid_nb))

        # 1 Corrupted notebook
        (tmp_path / "02_corrupted.ipynb").write_text("{ unquoted_json: True ")

        repo_map = ne.walk_and_scan_directory(str(tmp_path))
        report, is_clean = ne.generate_batch_analysis_report(repo_map, frozen_env, pkg_dist_map, None)

        assert is_clean is False
        assert len(repo_map.parse_errors) == 1
        assert "02_corrupted.ipynb" in str(repo_map.parse_errors[0].path)

    def test_missing_cells_array_handled_as_corrupted(self, tmp_path, mock_batch_env):
        """JSON file lacking a 'cells' array should be logged as an unparseable structure."""
        bad_schema = {"metadata": {"kernelspec": {"language": "python"}}}  # Missing 'cells'
        (tmp_path / "no_cells.ipynb").write_text(json.dumps(bad_schema))

        repo_map = ne.walk_and_scan_directory(str(tmp_path))
        assert len(repo_map.parse_errors) == 1
        # Asserts structural classification and error string non-emptiness
        assert repo_map.parse_errors[0].lang_label in ("corrupted", "error")
        err_msg = repo_map.parse_errors[0].parse_error.lower()
        assert "cells" in err_msg or "unparseable" in err_msg

    def test_hidden_and_checkpoint_dirs_ignored(self, tmp_path, mock_batch_env):
        """Files inside .ipynb_checkpoints, .git, or venv should never be scanned."""
        frozen_env, pkg_dist_map = mock_batch_env

        root_nb = {
            "metadata": {"kernelspec": {"language": "python"}},
            "cells": [{"cell_type": "code", "source": ["import math\n"]}]
        }
        (tmp_path / "root.ipynb").write_text(json.dumps(root_nb))

        checkpoint_dir = tmp_path / ".ipynb_checkpoints"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "root-checkpoint.ipynb").write_text("{ corrupted checkpoint json ")

        venv_dir = tmp_path / "venv" / "lib"
        venv_dir.mkdir(parents=True)
        (venv_dir / "ignored.ipynb").write_text("{ corrupted venv json ")

        repo_map = ne.walk_and_scan_directory(str(tmp_path))

        assert len(repo_map.parse_errors) == 0
        assert len(repo_map.scan_results) == 1
        assert repo_map.scan_results[0].path.name == "root.ipynb"

    def test_non_python_kernels_skipped_without_error(self, tmp_path, mock_batch_env):
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

        (tmp_path / "script.r.ipynb").write_text(json.dumps(r_nb))
        (tmp_path / "script.jl.ipynb").write_text(json.dumps(julia_nb))
        (tmp_path / "conflict.ipynb").write_text(json.dumps(conflict_nb))

        repo_map = ne.walk_and_scan_directory(str(tmp_path))
        report, is_clean = ne.generate_batch_analysis_report(repo_map, frozen_env, pkg_dist_map, None)

        assert is_clean is True
        assert len(repo_map.scan_results) == 0
        assert len(repo_map.non_python_files) == 3

    def test_cli_batch_aborts_on_parse_errors(self, tmp_path, monkeypatch, capsys):
        """Calling --batch --universal on a directory with parse errors must exit with code 1."""
        (tmp_path / "broken.ipynb").write_text("{ invalid json ")

        monkeypatch.setattr(sys, "argv", ["notebook_env.py", "--batch", str(tmp_path), "--universal"])

        with pytest.raises(SystemExit) as excinfo:
            ne.main()

        assert excinfo.value.code == 1

    def test_multiple_extra_index_urls_aggregated(self, tmp_path, mock_batch_env):
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

        (tmp_path / "01.ipynb").write_text(json.dumps(nb1))
        (tmp_path / "02.ipynb").write_text(json.dumps(nb2))

        repo_map = ne.walk_and_scan_directory(str(tmp_path))
        manifest = ne.generate_universal_manifest(repo_map, frozen_env, pkg_dist_map)

        assert "--extra-index-url https://index.a.com" in manifest
        assert "--extra-index-url https://index.b.com" in manifest
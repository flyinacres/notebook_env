"""
Unit and Integration Tests for Machine-Readable JSON Output (--format json).
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any

import pytest
import notebook_env as ne


@pytest.fixture
def mock_json_env(monkeypatch):
    frozen_env = {
        "numpy": "numpy==1.26.4",
        "pandas": "pandas==2.2.1",
        "torch": "torch==2.3.1+cu121",
        "scikit-learn": "scikit-learn==1.4.2",
    }
    raw_freeze = list(frozen_env.values())
    pkg_dist_map = {
        "numpy": ["numpy"],
        "pandas": ["pandas"],
        "torch": ["torch"],
        "sklearn": ["scikit-learn"],
    }
    monkeypatch.setattr(ne, "get_installed_environment", lambda: (frozen_env, raw_freeze))
    return frozen_env, pkg_dist_map


class TestSingleFileJsonOutput:
    """Tests single-file analysis JSON output shape, fields, and types."""

    def test_single_file_json_valid_contract(self, tmp_path: Path, mock_json_env, capsys: pytest.CaptureFixture[str], monkeypatch):
        nb_data = {
            "metadata": {"kernelspec": {"language": "python"}},
            "cells": [
                {"cell_type": "code", "execution_count": 1, "source": ["import pandas as pd\nimport numpy as np\n"]}
            ]
        }
        nb_path = tmp_path / "sample.ipynb"
        nb_path.write_text(json.dumps(nb_data), encoding="utf-8")

        monkeypatch.setattr(sys, "argv", ["notebook_env.py", str(nb_path), "--format", "json"])
        ne.main()

        captured = capsys.readouterr()
        # Ensure stdout is strictly valid JSON
        payload: Dict[str, Any] = json.loads(captured.out)

        assert payload["schema_version"] == "1.0"
        assert payload["tool_version"] == "43"
        assert payload["mode"] == "single_file"
        assert payload["notebook_path"] == str(nb_path)
        assert payload["is_python"] is True
        assert payload["parse_error"] is None

        # Verify dependencies structure
        dep_names = {d["name"]: d for d in payload["dependencies"]}
        assert "pandas" in dep_names
        assert dep_names["pandas"]["version"] == "2.2.1"
        assert dep_names["pandas"]["source"] == "import"
        assert dep_names["pandas"]["status"] == "pinned"
        assert dep_names["pandas"]["hardware_tagged"] is False

        assert "numpy" in dep_names
        assert dep_names["numpy"]["version"] == "1.26.4"

    def test_single_file_json_tracks_guarded_and_hardware_tags(self, tmp_path: Path, mock_json_env, capsys: pytest.CaptureFixture[str], monkeypatch):
        nb_data = {
            "metadata": {"kernelspec": {"language": "python"}},
            "cells": [
                {"cell_type": "code", "execution_count": 1, "source": [
                    "!pip install torch==2.3.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121\n",
                    "try:\n    import cupy\nexcept ImportError:\n    pass\n"
                ]}
            ]
        }
        nb_path = tmp_path / "gpu_guarded.ipynb"
        nb_path.write_text(json.dumps(nb_data), encoding="utf-8")

        monkeypatch.setattr(sys, "argv", ["notebook_env.py", str(nb_path), "--format", "json"])
        ne.main()

        captured = capsys.readouterr()
        payload = json.loads(captured.out)

        dep_map = {d["name"]: d for d in payload["dependencies"]}

        # Check torch hardware tag & scoped flags
        torch_dep = dep_map["torch"]
        assert torch_dep["version"] == "2.3.1+cu121"
        assert torch_dep["hardware_tagged"] is True
        assert torch_dep["source"] == "pip_command"
        assert "--extra-index-url" in torch_dep["flags"]

        # Check cupy guarded status and null version (not in active env)
        cupy_dep = dep_map["cupy"]
        assert cupy_dep["status"] == "guarded"
        assert cupy_dep["version"] is None
        assert cupy_dep["comment"] is not None

    def test_single_file_json_records_artifacts_written(self, tmp_path: Path, mock_json_env, capsys: pytest.CaptureFixture[str], monkeypatch):
        nb_data = {
            "metadata": {"kernelspec": {"language": "python"}},
            "cells": [{"cell_type": "code", "execution_count": 1, "source": ["import numpy\n"]}]
        }
        nb_path = tmp_path / "write_test.ipynb"
        nb_path.write_text(json.dumps(nb_data), encoding="utf-8")

        monkeypatch.setattr(sys, "argv", ["notebook_env.py", str(nb_path), "--output", "--format", "json"])
        ne.main()

        captured = capsys.readouterr()
        payload = json.loads(captured.out)

        assert payload["artifacts_written"] is not None
        assert "locked_notebook" in payload["artifacts_written"]
        written_file = Path(payload["artifacts_written"]["locked_notebook"])
        assert written_file.exists()
        assert written_file.name == "write_test_merged.ipynb"


class TestBatchJsonOutput:
    """Tests batch analysis JSON aggregation and individual notebook report folding."""

    def test_batch_json_aggregation_and_notebooks_list(self, tmp_path: Path, mock_json_env, capsys: pytest.CaptureFixture[str], monkeypatch):
        # Notebook 1: Standard python
        nb1 = {
            "metadata": {"kernelspec": {"language": "python"}},
            "cells": [{"cell_type": "code", "execution_count": 1, "source": ["import pandas\nimport missing_lib\n"]}]
        }
        (tmp_path / "01_first.ipynb").write_text(json.dumps(nb1), encoding="utf-8")

        # Notebook 2: R language
        nb2 = {
            "metadata": {"kernelspec": {"language": "R"}},
            "cells": []
        }
        (tmp_path / "02_r.ipynb").write_text(json.dumps(nb2), encoding="utf-8")

        monkeypatch.setattr(sys, "argv", ["notebook_env.py", "--batch", str(tmp_path), "--format", "json"])
        ne.main()

        captured = capsys.readouterr()
        payload = json.loads(captured.out)

        assert payload["schema_version"] == "1.0"
        assert payload["tool_version"] == "43"
        assert payload["mode"] == "batch"
        assert payload["target_dir"] == str(tmp_path)

        summary = payload["summary"]
        assert summary["is_clean"] is True
        assert summary["total_python_notebooks"] == 1
        assert summary["non_python_count"] == 1
        assert summary["non_python_languages"] == {"r": 1}
        assert "pandas" in summary["matched_packages"]
        assert "missing-lib" in summary["missing_packages"]
        assert summary["missing_packages"]["missing-lib"] == ["01_first.ipynb"]

        # Ensure notebooks[] array contains detailed individual report
        assert len(payload["notebooks"]) == 1
        nb_report = payload["notebooks"][0]
        assert nb_report["notebook_path"] == str(tmp_path / "01_first.ipynb")
        assert nb_report["is_python"] is True
        assert any(d["name"] == "pandas" for d in nb_report["dependencies"])

    def test_batch_json_records_multiple_written_artifacts(self, tmp_path: Path, mock_json_env, capsys: pytest.CaptureFixture[str], monkeypatch):
        nb = {
            "metadata": {"kernelspec": {"language": "python"}},
            "cells": [{"cell_type": "code", "execution_count": 1, "source": ["import numpy\n"]}]
        }
        (tmp_path / "proc.ipynb").write_text(json.dumps(nb), encoding="utf-8")

        monkeypatch.setattr(
            sys, "argv",
            ["notebook_env.py", "--batch", str(tmp_path), "--universal", "--output", "--format", "json"]
        )
        ne.main()

        captured = capsys.readouterr()
        payload = json.loads(captured.out)

        artifacts = payload["artifacts_written"]
        assert artifacts is not None
        assert "universal_manifest" in artifacts
        assert Path(artifacts["universal_manifest"]).exists()
        assert len(artifacts["locked_notebooks"]) == 1
        assert Path(artifacts["locked_notebooks"][0]).exists()


class TestStreamAndErrorHygiene:
    """Verifies stream separation and error handling under --format json."""

    def test_json_stdout_isolated_from_stderr_logs(self, tmp_path: Path, mock_json_env, capsys: pytest.CaptureFixture[str], monkeypatch):
        """Diagnostic warnings must go to stderr without polluting stdout JSON."""
        nb_data = {
            "metadata": {"kernelspec": {"language": "python"}},
            "cells": [
                {"cell_type": "code", "execution_count": 1, "source": [
                    "import importlib\n",
                    "pkg_var = 'dynamic_pkg'\n",
                    "mod = importlib.import_module(pkg_var)\n"
                ]}
            ]
        }
        nb_path = tmp_path / "dynamic_test.ipynb"
        nb_path.write_text(json.dumps(nb_data), encoding="utf-8")

        monkeypatch.setattr(sys, "argv", ["notebook_env.py", str(nb_path), "--format", "json"])
        ne.main()

        captured = capsys.readouterr()
        # stdout must parse cleanly with zero trailing/leading text
        payload = json.loads(captured.out)
        assert len(payload["warnings"]) > 0
        assert payload["warnings"][0]["type"] == "dynamic_import"

    def test_batch_corrupted_notebook_reported_in_json_summary(self, tmp_path: Path, mock_json_env, capsys: pytest.CaptureFixture[str], monkeypatch):
        bad_nb = tmp_path / "corrupted.ipynb"
        bad_nb.write_text("{not valid json", encoding="utf-8")

        monkeypatch.setattr(sys, "argv", ["notebook_env.py", "--batch", str(tmp_path), "--format", "json"])
        ne.main()

        captured = capsys.readouterr()
        payload = json.loads(captured.out)

        assert payload["summary"]["is_clean"] is False
        assert len(payload["summary"]["parse_errors"]) == 1
        err = payload["summary"]["parse_errors"][0]
        assert err["path"] == str(bad_nb)
        assert "not valid JSON" in err["cause"]
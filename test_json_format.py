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


# =====================================================================
# RENDERER-ISOLATION UNIT TESTS
# No notebook files, no argv, no environment mocking, no extraction
# pipeline at all. These construct the report dataclasses by hand and
# call the renderer functions directly, so a failure here can only mean
# the renderer is wrong, never that upstream analysis is wrong.
# =====================================================================

class TestJsonRendererUnit:
    """Isolated tests against format_json_single_report / format_json_batch_report."""

    def test_dependency_shape_guarded_found_and_not_found(self):
        deps = [
            ne.DependencyEntry(name="torch", version="2.3.1+cu121", source="pip_command",
                                status="pinned", flags=["--extra-index-url", "https://x"]),
            ne.DependencyEntry(name="cupy", version="13.0.0", source="import", status="guarded",
                                is_comment=True, comment_text="optional or conditional dependency inside try/except block"),
            ne.DependencyEntry(name="cupy2", version="", source="import", status="guarded",
                                is_comment=True, comment_text="cupy2 (optional..., not found in active env)"),
            ne.DependencyEntry(name="xgboost", version="", source="import", status="pinned",
                                is_comment=True, comment_text="imported as 'xgboost', not currently found in active env"),
        ]
        report = ne.NotebookAnalysisReport(
            notebook_path="x.ipynb", is_python=True, lang_label="python", dependencies=deps
        )
        payload = json.loads(ne.format_json_single_report(report))
        by_name = {d["name"]: d for d in payload["dependencies"]}

        torch = by_name["torch"]
        assert torch["status"] == "pinned"
        assert torch["hardware_tagged"] is True
        assert torch["comment"] is None

        guarded_found = by_name["cupy"]
        assert guarded_found["status"] == "guarded"
        assert guarded_found["version"] == "13.0.0"
        assert guarded_found["comment"] is not None

        guarded_not_found = by_name["cupy2"]
        assert guarded_not_found["status"] == "guarded"
        assert guarded_not_found["version"] is None

        plain_not_found = by_name["xgboost"]
        assert plain_not_found["status"] == "pinned"
        assert plain_not_found["version"] is None

    def test_hardware_tagged_derived_purely_from_version(self):
        """hardware_tagged must come from the '+' in version, not a separately tracked flag."""
        tagged = ne.DependencyEntry(name="torch", version="2.3.1+cu121")
        untagged = ne.DependencyEntry(name="pandas", version="2.2.1")
        no_version = ne.DependencyEntry(name="xgboost", version="", is_comment=True, comment_text="# not found")

        assert tagged.to_report_dict()["hardware_tagged"] is True
        assert untagged.to_report_dict()["hardware_tagged"] is False
        assert no_version.to_report_dict()["hardware_tagged"] is False

    def test_warning_and_notice_location_fields(self):
        report = ne.NotebookAnalysisReport(
            notebook_path="x.ipynb", is_python=True, lang_label="python",
            warnings=[
                ne.DiagnosticEvent(type="dynamic_import", detail="d", cell_idx=3, line_idx=1, level="warning"),
                ne.DiagnosticEvent(type="missing_hardware_index", detail="d", level="warning"),
            ],
            notices=[
                ne.DiagnosticEvent(type="system_command", detail="d", cell_idx=0, line_idx=2, level="notice"),
            ],
        )
        payload = json.loads(ne.format_json_single_report(report))

        dyn = next(w for w in payload["warnings"] if w["type"] == "dynamic_import")
        assert dyn["cell_idx"] == 3 and dyn["line_idx"] == 1

        # Known, deliberate gap: DependencyEntry carries no provenance today, so this
        # warning type must render with null location rather than a fabricated value.
        hw = next(w for w in payload["warnings"] if w["type"] == "missing_hardware_index")
        assert hw["cell_idx"] is None and hw["line_idx"] is None

        notice = payload["notices"][0]
        assert notice["cell_idx"] == 0 and notice["line_idx"] == 2

    def test_warnings_and_notices_level_consistency(self):
        """Every item placed in `warnings` must be level='warning'; `notices` must be level='notice'."""
        report = ne.NotebookAnalysisReport(
            notebook_path="x.ipynb", is_python=True, lang_label="python",
            warnings=[ne.DiagnosticEvent(type="dynamic_import", detail="d", level="warning")],
            notices=[ne.DiagnosticEvent(type="system_command", detail="d", level="notice")],
        )
        assert all(w.level == "warning" for w in report.warnings)
        assert all(n.level == "notice" for n in report.notices)

    def test_gpu_serialization_present_and_null(self):
        gpu = ne.GpuInfo(has_gpu=True, active_framework="PyTorch", device_name="RTX 3090",
                          frameworks=["torch"], probe_errors=[])
        with_gpu = ne.NotebookAnalysisReport(notebook_path="x.ipynb", is_python=True, lang_label="python", gpu=gpu)
        without_gpu = ne.NotebookAnalysisReport(notebook_path="x.ipynb", is_python=True, lang_label="python", gpu=None)

        payload_with = json.loads(ne.format_json_single_report(with_gpu))
        payload_without = json.loads(ne.format_json_single_report(without_gpu))

        assert payload_with["gpu"] == {
            "has_gpu": True, "framework": "PyTorch", "device_name": "RTX 3090",
            "frameworks_detected": ["torch"], "probe_errors": []
        }
        assert payload_without["gpu"] is None

    def test_artifacts_written_passthrough(self):
        report = ne.NotebookAnalysisReport(notebook_path="x.ipynb", is_python=True, lang_label="python")

        assert json.loads(ne.format_json_single_report(report))["artifacts_written"] is None

        written = {"locked_notebook": "x_merged.ipynb"}
        assert json.loads(ne.format_json_single_report(report, artifacts_written=written))["artifacts_written"] == written

    def test_schema_and_tool_version_constants(self):
        report = ne.NotebookAnalysisReport(notebook_path="x.ipynb", is_python=True, lang_label="python")
        payload = json.loads(ne.format_json_single_report(report))
        assert payload["schema_version"] == ne.SCHEMA_VERSION
        assert payload["tool_version"] == ne.TOOL_VERSION
        assert isinstance(payload["tool_version"], str)

    def test_batch_summary_shape(self):
        nb_report = ne.NotebookAnalysisReport(
            notebook_path="repo/a.ipynb", is_python=True, lang_label="python",
            dependencies=[ne.DependencyEntry(name="torch", version="2.3.1+cu121", status="pinned")]
        )
        summary = ne.BatchAnalysisSummary(
            target_dir="./repo",
            total_python_notebooks=1,
            non_python_count=1,
            non_python_languages={"r": 1},
            parse_errors=[{"path": "repo/bad.ipynb", "cause": "File is not valid JSON."}],
            matched_packages={"numpy", "pandas"},
            missing_packages={"xgboost": ["a.ipynb"]},
            batch_hardware_warnings={"torch==2.3.1+cu121": ["a.ipynb"]},
            primary_url="https://download.pytorch.org/whl/cu121",
            primary_url_reason="Sole index URL harvested across batch (1 notebook(s))",
            notebooks=[nb_report],
        )
        payload = json.loads(ne.format_json_batch_report(summary))

        assert payload["mode"] == "batch"
        assert payload["target_dir"] == "./repo"
        s = payload["summary"]
        assert s["is_clean"] is False  # parse_errors present
        assert s["parse_errors"] == [{"path": "repo/bad.ipynb", "cause": "File is not valid JSON."}]
        assert s["matched_packages"] == sorted({"numpy", "pandas"})
        assert s["missing_packages"] == {"xgboost": ["a.ipynb"]}
        assert s["hardware_warnings"] == {"torch==2.3.1+cu121": ["a.ipynb"]}
        assert s["primary_index_url"] == "https://download.pytorch.org/whl/cu121"

        assert len(payload["notebooks"]) == 1
        assert payload["notebooks"][0]["notebook_path"] == "repo/a.ipynb"
        assert payload["notebooks"][0]["dependencies"][0]["name"] == "torch"


# =====================================================================
# CONSOLE / JSON PARITY
# Direct regression guard against the two-renderers-drift bug class:
# same report object in, both renderers must agree on the same facts.
# =====================================================================

class TestConsoleJsonParity:
    """Feeds one shared report object to both renderers and checks they agree."""

    def test_batch_console_and_json_agree_on_counts(self):
        nb_report = ne.NotebookAnalysisReport(notebook_path="repo/a.ipynb", is_python=True, lang_label="python")
        summary = ne.BatchAnalysisSummary(
            target_dir="./repo",
            total_python_notebooks=1,
            matched_packages={"numpy", "pandas", "torch"},
            missing_packages={"xgboost": ["a.ipynb"], "shap": ["a.ipynb"]},
            magic_warnings=[ne.DiagnosticEvent(type="external_requirement", detail="uses -r reqs.txt", level="warning")],
            magic_notices=[ne.DiagnosticEvent(type="system_command", detail="uses apt-get", level="notice")],
            promotions=[ne.PromotionDetail(import_name="umap.plot", promoted_name="umap-learn[plot]",
                                            version="0.5.5", detail="umap.plot -> umap-learn[plot]==0.5.5")],
            primary_url="https://download.pytorch.org/whl/cu121",
            primary_url_reason="Sole index URL harvested across batch (1 notebook(s))",
            notebooks=[nb_report],
        )

        console_text = ne.format_console_report(summary)
        json_payload = json.loads(ne.format_json_batch_report(summary))

        # Matched/missing package counts must agree between renderers.
        assert json_payload["summary"]["total_python_notebooks"] == 1
        assert f"Installed & Verified: {len(json_payload['summary']['matched_packages'])} packages" in console_text
        assert f"Packages missing from current environment: {len(json_payload['summary']['missing_packages'])}" in console_text

        # Every warning/notice/promotion detail present in JSON must also appear in console text.
        for w in summary.magic_warnings:
            assert w.detail in console_text
        for n in summary.magic_notices:
            assert n.detail in console_text
        for p in summary.promotions:
            assert p.detail in console_text

        assert json_payload["summary"]["primary_index_url"] == summary.primary_url
        assert summary.primary_url in console_text

    def test_single_file_console_and_json_agree_on_warning_count(
        self, tmp_path: Path, mock_json_env, capsys: pytest.CaptureFixture[str], monkeypatch, caplog
    ):
        """Single-file text mode has no standalone formatter (still ad-hoc logger calls),
        so parity here is checked by running the pipeline twice and comparing the
        warning count logged to stderr against the warning count in the JSON payload."""
        nb_data = {
            "metadata": {"kernelspec": {"language": "python"}},
            "cells": [
                {"cell_type": "code", "execution_count": 1, "source": [
                    "!pip install torch==2.3.1+cu121\n",  # hardware tag, no index url -> warning
                    "import importlib\n",
                    "pkg_var = 'x'\n",
                    "mod = importlib.import_module(pkg_var)\n"  # dynamic import -> warning
                ]}
            ]
        }
        nb_path = tmp_path / "warn_test.ipynb"
        nb_path.write_text(json.dumps(nb_data), encoding="utf-8")

        import logging
        caplog.set_level(logging.WARNING, logger="notebook_env")

        monkeypatch.setattr(sys, "argv", ["notebook_env.py", str(nb_path), "--format", "text"])
        ne.main()
        text_warning_lines = [r.message for r in caplog.records if r.message.strip().startswith("•")]
        capsys.readouterr()  # drain stdout from the text-mode run before the json-mode run below

        caplog.clear()
        monkeypatch.setattr(sys, "argv", ["notebook_env.py", str(nb_path), "--format", "json"])
        ne.main()
        json_payload = json.loads(capsys.readouterr().out)

        assert len(text_warning_lines) == len(json_payload["warnings"])
        assert len(json_payload["warnings"]) == 2

    def test_missing_hardware_index_has_null_location_end_to_end(
        self, tmp_path: Path, mock_json_env, capsys: pytest.CaptureFixture[str], monkeypatch
    ):
        """Runs the real pipeline (not a hand-built object) to guard the known
        DependencyEntry-has-no-provenance gap: a hardware-tag warning must come out
        with a null cell_idx/line_idx, never a fabricated one, until DependencyEntry
        gains real provenance tracking."""
        nb_data = {
            "metadata": {"kernelspec": {"language": "python"}},
            "cells": [
                {"cell_type": "code", "execution_count": 1, "source": ["!pip install torch==2.3.1+cu121\n"]}
            ]
        }
        nb_path = tmp_path / "hw_tag.ipynb"
        nb_path.write_text(json.dumps(nb_data), encoding="utf-8")

        monkeypatch.setattr(sys, "argv", ["notebook_env.py", str(nb_path), "--format", "json"])
        ne.main()
        payload = json.loads(capsys.readouterr().out)

        hw_warnings = [w for w in payload["warnings"] if w["type"] == "missing_hardware_index"]
        assert len(hw_warnings) == 1
        assert hw_warnings[0]["cell_idx"] is None
        assert hw_warnings[0]["line_idx"] is None
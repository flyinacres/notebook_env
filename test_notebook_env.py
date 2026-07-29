"""
Tests for notebook_env.py (v19).

Assumes the tool module is importable as `notebook_env`. Adjust the import
below if the actual filename differs.

Tiers, per the test architecture doc:
  - Unit tests: pure logic, mocked subprocess/importlib/hardware libs
  - Integration/fixture tests: synthetic .ipynb dicts, mocked IPython history
  - E2E smoke tests (real GPU/CPU containers) are NOT included here;
    they require actual hardware and are out of scope for this file.
"""

import json
import sys
import types
import pytest

import notebook_env as ne


# =====================================================================
# 1. AST Parsing & Import Harvesting
# =====================================================================

class TestImportExtraction:
    def test_standard_imports(self):
        sources = ["import numpy as np\nimport os, sys\nfrom sklearn.model_selection import train_test_split"]
        imports, _ = ne.extract_imports_from_sources(sources)
        assert "numpy" in imports
        assert "os" in imports
        assert "sys" in imports
        assert "sklearn" in imports

    def test_deep_submodule_import_resolves_top_level(self):
        sources = ["import torch.nn.functional as F"]
        imports, submodules = ne.extract_imports_from_sources(sources)
        assert "torch" in imports
        assert "torch.nn.functional" in submodules.get("torch", set())

    def test_magics_and_shell_escapes_stripped(self):
        sources = ["%matplotlib inline\n%%writefile foo.py\n!pip install foo\nimport pandas as pd"]
        imports, _ = ne.extract_imports_from_sources(sources)
        assert "pandas" in imports

    def test_syntax_error_in_one_cell_does_not_block_others(self):
        sources = [
            "import pandas as pd",
            "def foo(",  # broken syntax
            "import requests",
        ]
        imports, _ = ne.extract_imports_from_sources(sources)
        assert "pandas" in imports
        assert "requests" in imports

    def test_empty_and_non_code_sources_return_empty(self):
        imports, submodules = ne.extract_imports_from_sources([])
        assert imports == set()
        assert submodules == {}

    def test_commented_import_not_extracted(self):
        sources = ["# import tensorflow as tf\nimport json"]
        imports, _ = ne.extract_imports_from_sources(sources)
        assert "tensorflow" not in imports
        # json is stdlib, but this only tests extraction, not filtering
        assert "json" in imports


# =====================================================================
# 2. Index URL Harvesting
# =====================================================================

class TestIndexUrlHarvesting:
    def test_extra_index_url_flag(self):
        sources = ["!pip install torch --extra-index-url https://download.pytorch.org/whl/cu121"]
        urls = ne.harvest_index_urls_from_sources(sources)
        assert "https://download.pytorch.org/whl/cu121" in urls

    def test_short_flag(self):
        sources = ["!pip install -i https://pypi.org/simple somepkg"]
        urls = ne.harvest_index_urls_from_sources(sources)
        assert "https://pypi.org/simple" in urls

    def test_quoted_and_multiple_urls_across_cells(self):
        sources = [
            "!pip install foo --extra-index-url 'https://a.example.com'",
            '!pip install bar --extra-index-url "https://b.example.com"',
        ]
        urls = ne.harvest_index_urls_from_sources(sources)
        assert "https://a.example.com" in urls
        assert "https://b.example.com" in urls

    def test_malformed_flag_no_url_not_captured(self):
        sources = ["!pip install foo --extra-index-url"]
        urls = ne.harvest_index_urls_from_sources(sources)
        assert urls == set()

    def test_commented_out_pip_call_not_harvested(self):
        sources = ["# !pip install torch --extra-index-url https://download.pytorch.org/whl/cu121"]
        urls = ne.harvest_index_urls_from_sources(sources)
        assert urls == set()


# =====================================================================
# 3. Dual-Path Ingestion Engine
# =====================================================================

class TestDualPathIngestion:
    def test_path_a_reads_saved_notebook(self, tmp_path):
        nb = {
            "cells": [
                {"cell_type": "code", "source": ["import numpy as np\n"]},
                {"cell_type": "markdown", "source": ["# not code\n"]},
            ]
        }
        nb_path = tmp_path / "test.ipynb"
        nb_path.write_text(json.dumps(nb))

        success, imports, submodules, code_sources, error_msg = ne.extract_from_file(str(nb_path))
        assert success is True
        assert error_msg is None
        assert "numpy" in imports
        assert len(code_sources) == 1

    def test_path_a_missing_file_returns_error(self):
        success, imports, submodules, code_sources, error_msg = ne.extract_from_file("does_not_exist.ipynb")
        assert success is False
        assert imports == set()
        assert "not found" in error_msg.lower()

    def test_path_a_corrupted_json_returns_error(self, tmp_path):
        bad_path = tmp_path / "bad.ipynb"
        bad_path.write_text("{not valid json")

        success, imports, submodules, code_sources, error_msg = ne.extract_from_file(str(bad_path))
        assert success is False
        assert "not a valid" in error_msg.lower()

    def test_main_exits_on_missing_file(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["notebook_env.py", "does_not_exist.ipynb"])
        with pytest.raises(SystemExit) as exc:
            ne.main()
        assert exc.value.code == 1
        captured = capsys.readouterr()
        assert "not found" in captured.out.lower() or "error" in captured.out.lower()

    def test_main_exits_on_corrupted_json(self, tmp_path, monkeypatch, capsys):
        bad_path = tmp_path / "bad.ipynb"
        bad_path.write_text("{not valid json")
        monkeypatch.setattr(sys, "argv", ["notebook_env.py", str(bad_path)])

        with pytest.raises(SystemExit) as exc:
            ne.main()
        assert exc.value.code == 1
        captured = capsys.readouterr()
        assert "not a valid" in captured.out.lower() or "error" in captured.out.lower()

    def test_path_b_reads_live_session_history(self, monkeypatch):
        fake_main = types.ModuleType("__main__")
        fake_main.In = ["", "import requests", "import pandas as pd"]
        monkeypatch.setitem(sys.modules, "__main__", fake_main)

        imports, submodules, code_sources = ne.extract_from_active_session()
        assert "requests" in imports
        assert "pandas" in imports

    def test_path_b_empty_history_returns_empty_manifest(self, monkeypatch):
        fake_main = types.ModuleType("__main__")
        fake_main.In = []
        monkeypatch.setitem(sys.modules, "__main__", fake_main)

        imports, submodules, code_sources = ne.extract_from_active_session()
        assert imports == set()
        assert code_sources == []


# =====================================================================
# 4. Hardware Acceleration Inspection
# =====================================================================

def _install_fake_module(monkeypatch, name, module):
    monkeypatch.setitem(sys.modules, name, module)


class TestGpuInspection:
    def test_no_frameworks_imported_skips_check(self):
        result = ne.inspect_gpu_environment({"pandas", "requests"})
        assert result is None

    def test_torch_cuda_active(self, monkeypatch):
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda i: "NVIDIA GeForce RTX 3090",
        )
        fake_torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
        _install_fake_module(monkeypatch, "torch", fake_torch)

        result = ne.inspect_gpu_environment({"torch"})
        assert result["has_gpu"] is True
        assert result["active_framework"] == "PyTorch"
        assert "RTX 3090" in result["device_name"]

    def test_torch_mps_active(self, monkeypatch):
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        fake_torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True))
        _install_fake_module(monkeypatch, "torch", fake_torch)

        result = ne.inspect_gpu_environment({"torch"})
        assert result["has_gpu"] is True
        assert "Metal" in result["device_name"]

    def test_torch_imported_no_gpu_available(self, monkeypatch):
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        fake_torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
        _install_fake_module(monkeypatch, "torch", fake_torch)

        result = ne.inspect_gpu_environment({"torch"})
        assert result["has_gpu"] is False
        assert result["frameworks"] == ["torch"]

    def test_tensorflow_gpu_active(self, monkeypatch):
        fake_tf = types.ModuleType("tensorflow")
        fake_gpu_device = object()
        fake_tf.config = types.SimpleNamespace(
            list_physical_devices=lambda kind: [fake_gpu_device] if kind == "GPU" else [],
            experimental=types.SimpleNamespace(
                get_device_details=lambda d: {"device_name": "Tesla T4"}
            ),
        )
        _install_fake_module(monkeypatch, "tensorflow", fake_tf)

        result = ne.inspect_gpu_environment({"tensorflow"})
        assert result["has_gpu"] is True
        assert result["active_framework"] == "TensorFlow"
        assert "Tesla T4" in result["device_name"]

    def test_jax_accelerator_active(self, monkeypatch):
        fake_jax = types.ModuleType("jax")
        fake_device = types.SimpleNamespace(platform="gpu", device_kind="A100")
        fake_jax.devices = lambda: [fake_device]
        _install_fake_module(monkeypatch, "jax", fake_jax)

        result = ne.inspect_gpu_environment({"jax"})
        assert result["has_gpu"] is True
        assert result["active_framework"] == "JAX"

    def test_framework_not_installed_falls_back_gracefully(self, monkeypatch):
        # Ensure "torch" is not importable
        monkeypatch.setitem(sys.modules, "torch", None)
        result = ne.inspect_gpu_environment({"torch"})
        assert result["has_gpu"] is False


# =====================================================================
# 5. Requirement Correlation & Blueprint Generation
# =====================================================================

class TestPackageRequirements:
    def test_local_tag_without_harvested_url_warns(self):
        manifest, tagged, warnings = ne.process_package_requirements(["torch==2.3.1+cu121"], set())
        assert "torch==2.3.1+cu121" in warnings
        assert tagged == [("torch==2.3.1+cu121", [])]

    def test_local_tag_with_harvested_url_no_warning(self):
        manifest, tagged, warnings = ne.process_package_requirements(
            ["torch==2.3.1+cu121"], {"https://download.pytorch.org/whl/cu121"}
        )
        assert warnings == []
        assert "--extra-index-url https://download.pytorch.org/whl/cu121" in manifest
        assert tagged[0][0] == "torch==2.3.1+cu121"
        assert "https://download.pytorch.org/whl/cu121" in tagged[0][1]

    def test_uninstalled_top_level_import_placeholder(self):
        # Simulates main()'s handling when a package isn't found in pip freeze
        pinned_manifest = ["# some_unknown_pkg (imported as 'some_unknown_pkg', not currently found in active env)"]
        manifest, tagged, warnings = ne.process_package_requirements(pinned_manifest, set())
        assert manifest == pinned_manifest
        assert tagged == []
        assert warnings == []


class TestBlueprintGeneration:
    def test_returns_both_sections(self):
        blueprint = ne.generate_production_blueprint(["numpy==1.26.0"])
        assert "step1_markdown" in blueprint
        assert "step2_code" in blueprint

    def test_python_version_guard_matches_runtime(self):
        blueprint = ne.generate_production_blueprint(["numpy==1.26.0"])
        expected = f"REQUIRED_PYTHON = ({sys.version_info.major}, {sys.version_info.minor})"
        assert expected in blueprint["step2_code"]

    def test_gpu_section_included_when_gpu_present(self):
        gpu_info = {
            "has_gpu": True,
            "active_framework": "PyTorch",
            "device_name": "NVIDIA GeForce RTX 3090 (via PyTorch)",
            "frameworks": ["torch"],
        }
        blueprint = ne.generate_production_blueprint(["torch==2.3.1"], gpu_info=gpu_info)
        assert "Hardware Acceleration" in blueprint["step1_markdown"]
        assert "RTX 3090" in blueprint["step1_markdown"]

    def test_gpu_section_omitted_when_no_gpu(self):
        blueprint = ne.generate_production_blueprint(["numpy==1.26.0"], gpu_info=None)
        assert "Hardware Acceleration" not in blueprint["step1_markdown"]

    def test_full_freeze_appended_after_manifest(self):
        blueprint = ne.generate_production_blueprint(
            ["numpy==1.26.0"], full_freeze_lines=["# certifi==2024.2.2"]
        )
        code = blueprint["step2_code"]
        manifest_pos = code.find("numpy==1.26.0")
        freeze_pos = code.find("certifi==2024.2.2")
        assert manifest_pos != -1 and freeze_pos != -1
        assert manifest_pos < freeze_pos


# =====================================================================
# 6. Runtime Execution (Cell 2 payload actually runs and writes the file)
# =====================================================================

class TestRuntimeExecution:
    def test_generated_code_writes_pinned_requirements_file(self, tmp_path, monkeypatch):
        manifest = ["numpy==1.26.0", "pandas==2.2.1"]
        blueprint = ne.generate_production_blueprint(manifest)

        monkeypatch.chdir(tmp_path)

        import subprocess
        monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: types.SimpleNamespace(returncode=0))

        exec(compile(blueprint["step2_code"], "<string>", "exec"), {"__builtins__": __builtins__})

        req_file = tmp_path / "pinned_requirements.txt"
        assert req_file.exists(), "Cell 2 code executed but failed to write pinned_requirements.txt"

        content = req_file.read_text(encoding="utf-8")
        assert "numpy==1.26.0" in content
        assert "pandas==2.2.1" in content


# =====================================================================
# 7. Full Pipeline Detection Accuracy (exercises main() end to end)
# =====================================================================

class TestEndToEndDetectionAccuracy:
    def test_full_detection_pipeline_output_reflects_real_detection(self, tmp_path, monkeypatch, capsys):
        """
        Runs the actual pipeline: AST extraction -> stdlib filtering -> pypi
        name mapping -> environment correlation -> blueprint generation,
        all inside main(). Nothing here hand-types the expected manifest;
        the assertions check main()'s real printed output.
        """
        nb = {
            "cells": [
                {"cell_type": "code", "source": ["import numpy as np\n", "from sklearn.ensemble import RandomForestClassifier\n"]},
                {"cell_type": "code", "source": ["# import tensorflow as tf\n", "%matplotlib inline\n", "import cv2\n"]},
            ]
        }
        nb_path = tmp_path / "test_detection.ipynb"
        nb_path.write_text(json.dumps(nb), encoding="utf-8")

        # Only source of truth for correlation - nothing else is hand-typed.
        mock_frozen_env = {
            "numpy": "numpy==1.26.4",
            "scikit-learn": "scikit-learn==1.4.2",
            "opencv-python": "opencv-python==4.9.0.80",
        }
        mock_raw_freeze = ["numpy==1.26.4", "scikit-learn==1.4.2", "opencv-python==4.9.0.80"]

        monkeypatch.setattr(ne, "get_installed_environment", lambda: (mock_frozen_env, mock_raw_freeze))
        monkeypatch.setattr(ne, "resolve_opencv_variant", lambda submodules=None: "opencv-python")
        monkeypatch.chdir(tmp_path)

        import subprocess
        monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: types.SimpleNamespace(returncode=0))
        monkeypatch.setattr(sys, "argv", ["notebook_env.py", str(nb_path)])

        ne.main()
        captured = capsys.readouterr()

        # Detected via the real AST scan + real IMPORT_TO_PYPI_MAP + real correlation
        assert "numpy==1.26.4" in captured.out
        assert "scikit-learn==1.4.2" in captured.out
        assert "opencv-python==4.9.0.80" in captured.out

        # Filtered out correctly: commented-out import, magic command
        assert "tensorflow" not in captured.out
        assert "matplotlib" not in captured.out
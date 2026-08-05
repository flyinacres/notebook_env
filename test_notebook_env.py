"""
Tests for notebook_env.py (v26).

Unit and integration test suite exercising AST parsing, index URL harvesting,
dynamic package/extras resolution, dual-path ingestion, GPU inspection,
blueprint generation, and runtime sandbox execution.
"""

import json
import sys
import types
import importlib.metadata
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

import pytest

import notebook_env as ne
from notebook_env import (
    StatusLabel,
    GpuInfo,
    BlueprintResult,
)


# =====================================================================
# 1. AST PARSING & IMPORT HARVESTING
# =====================================================================

class TestImportExtraction:
    """Tests raw AST extraction of top-level imports and submodules from source code strings."""

    def test_standard_imports(self) -> None:
        sources: List[str] = [
            "import numpy as np\nimport os, sys\nfrom sklearn.model_selection import train_test_split"
        ]
        # Unpack 3-tuple
        imports, _, _ = ne.extract_imports_from_sources(sources)
        assert "numpy" in imports
        assert "os" in imports
        assert "sys" in imports
        assert "sklearn" in imports

    def test_deep_submodule_import_resolves_top_level(self) -> None:
        sources: List[str] = ["import torch.nn.functional as F"]
        # Unpack 3-tuple
        imports, submodules, _ = ne.extract_imports_from_sources(sources)
        assert "torch" in imports
        assert "torch.nn.functional" in submodules.get("torch", set())

    def test_magics_and_shell_escapes_stripped(self) -> None:
        sources: List[str] = [
            "%matplotlib inline\n%%writefile foo.py\n!pip install foo\nimport pandas as pd"
        ]
        # Unpack 3-tuple
        imports, _, _ = ne.extract_imports_from_sources(sources)
        assert "pandas" in imports

    def test_syntax_error_in_one_cell_does_not_block_others(self) -> None:
        sources: List[str] = [
            "import pandas as pd",
            "def foo(",
            "import requests",
        ]
        # Unpack 3-tuple
        imports, _, _ = ne.extract_imports_from_sources(sources)
        assert "pandas" in imports
        assert "requests" in imports

    def test_empty_and_non_code_sources_return_empty(self) -> None:
        # Unpack 3-tuple
        imports, submodules, guarded = ne.extract_imports_from_sources([])
        assert imports == set()
        assert submodules == {}
        assert guarded == set()

    def test_commented_import_not_extracted(self) -> None:
        sources: List[str] = ["# import tensorflow as tf\nimport json"]
        # Unpack 3-tuple
        imports, _, _ = ne.extract_imports_from_sources(sources)
        assert "tensorflow" not in imports
        assert "json" in imports

class TestGuardedImports:
    """Tests detection of guarded imports inside try/except and if/else blocks."""

    def test_try_except_import_marked_as_guarded(self) -> None:
        """Imports inside try/except blocks must be marked as guarded."""
        # Arrange
        sources = [
            "import numpy as np\n"
            "try:\n"
            "    import cupy as cp\n"
            "except ImportError:\n"
            "    pass"
        ]

        # Act
        imports, submodules, guarded_imports = ne.extract_imports_from_sources(sources)

        # Assert: numpy is unconditional; cupy is extracted as guarded
        assert "numpy" in imports
        assert "numpy" not in guarded_imports
        assert "cupy" in guarded_imports

    def test_if_statement_import_marked_as_guarded(self) -> None:
        """Imports inside if/elif/else conditionals must be marked as guarded."""
        # Arrange
        sources = [
            "import os\n"
            "if sys.platform == 'win32':\n"
            "    import pywin32\n"
        ]

        # Act
        imports, submodules, guarded_imports = ne.extract_imports_from_sources(sources)

        # Assert: os is unconditional; pywin32 is extracted as guarded
        assert "os" in imports
        assert "os" not in guarded_imports
        assert "pywin32" in guarded_imports

    def test_uninstalled_guarded_import_formatted_as_optional_in_manifest(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Uninstalled guarded imports should be formatted as optional comments in the manifest."""
        # Arrange
        frozen_env = {"numpy": "numpy==1.26.4"}
        submodules = {}
        imports = {"numpy", "cupy"}
        guarded_imports = {"cupy"}

        # Act: Request manifest entries passing guarded_imports
        pinned_entries, notices = ne.build_manifest_entries(
            imports, submodules, frozen_env, guarded_imports=guarded_imports
        )

        # Assert: cupy should be annotated as an optional/guarded fallback comment
        cupy_pin = next((p for p in pinned_entries if "cupy" in p), "")
        assert cupy_pin.startswith("#")
        assert "guarded" in cupy_pin.lower() or "optional" in cupy_pin.lower()

# =====================================================================
# 2. INDEX URL HARVESTING
# =====================================================================

class TestIndexUrlHarvesting:
    """Tests extraction of custom PyPI index URLs (--extra-index-url, -i) from notebook cells."""

    def test_extra_index_url_flag(self) -> None:
        # Arrange
        sources: List[str] = ["!pip install torch --extra-index-url https://download.pytorch.org/whl/cu121"]

        # Act
        urls: Set[str] = ne.harvest_index_urls_from_sources(sources)

        # Assert
        assert "https://download.pytorch.org/whl/cu121" in urls

    def test_short_flag(self) -> None:
        # Arrange
        sources: List[str] = ["!pip install -i https://pypi.org/simple somepkg"]

        # Act
        urls: Set[str] = ne.harvest_index_urls_from_sources(sources)

        # Assert
        assert "https://pypi.org/simple" in urls

    def test_quoted_and_multiple_urls_across_cells(self) -> None:
        # Arrange
        sources: List[str] = [
            "!pip install foo --extra-index-url 'https://a.example.com'",
            '!pip install bar --extra-index-url "https://b.example.com"',
        ]

        # Act
        urls: Set[str] = ne.harvest_index_urls_from_sources(sources)

        # Assert
        assert "https://a.example.com" in urls
        assert "https://b.example.com" in urls

    def test_malformed_flag_no_url_not_captured(self) -> None:
        # Arrange: Flag provided without an accompanying URL
        sources: List[str] = ["!pip install foo --extra-index-url"]

        # Act
        urls: Set[str] = ne.harvest_index_urls_from_sources(sources)

        # Assert
        assert urls == set()

    def test_commented_out_pip_call_not_harvested(self) -> None:
        # Arrange
        sources: List[str] = ["# !pip install torch --extra-index-url https://download.pytorch.org/whl/cu121"]

        # Act
        urls: Set[str] = ne.harvest_index_urls_from_sources(sources)

        # Assert
        assert urls == set()


# =====================================================================
# 3. DYNAMIC RESOLUTION & PROVIDES-EXTRA PROMOTION
# =====================================================================

class TestDynamicResolution:
    """Tests PyPI package name correlation and extra dependency promotion (e.g., umap.plot -> umap-learn[plot])."""

    def test_submodule_import_promotes_to_package_extras(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange: Mock package distribution declaring 'Provides-Extra: plot'
        fake_dist = types.SimpleNamespace(
            metadata=types.SimpleNamespace(get_all=lambda key: ["plot"] if key == "Provides-Extra" else [])
        )
        monkeypatch.setattr(importlib.metadata, "distribution", lambda pkg: fake_dist)
        if hasattr(importlib.metadata, "packages_distributions"):
            monkeypatch.setattr(importlib.metadata, "packages_distributions", lambda: {"umap": ["umap-learn"]})

        frozen_env: Dict[str, str] = {"umap-learn": "umap-learn==0.5.5"}
        submodules_set: Set[str] = {"umap.plot"}

        # Act
        pin, notice = ne.resolve_pypi_package_and_extras("umap", submodules_set, frozen_env)

        # Assert
        assert pin == "umap-learn[plot]==0.5.5"
        assert notice is not None
        assert "umap-learn[plot]==0.5.5" in notice

    def test_fallback_map_used_when_uninstalled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange: Package absent from active environment
        if hasattr(importlib.metadata, "packages_distributions"):
            monkeypatch.setattr(importlib.metadata, "packages_distributions", lambda: {})

        frozen_env: Dict[str, str] = {}

        # Act
        pin, notice = ne.resolve_pypi_package_and_extras("sklearn", set(), frozen_env)

        # Assert
        assert pin.startswith("#")
        assert "scikit-learn" in pin
        assert "sklearn" in pin
        assert notice is None


# =====================================================================
# 4. DUAL-PATH INGESTION ENGINE
# =====================================================================

class TestDualPathIngestion:
    """Tests Path A (saved notebook file ingestion) and Path B (live kernel session ingestion)."""

    def test_path_a_reads_saved_notebook(self, tmp_path: Path) -> None:
        nb: Dict[str, Any] = {
            "metadata": {"kernelspec": {"language": "python"}},
            "cells": [
                {"cell_type": "code", "source": ["import numpy as np\n"]},
                {"cell_type": "markdown", "source": ["# not code\n"]},
            ]
        }
        nb_path: Path = tmp_path / "test.ipynb"
        nb_path.write_text(json.dumps(nb), encoding="utf-8")

        # Unpack 7-tuple
        success, imports, submodules, code_sources, err, lang_label, guarded = ne.extract_from_file(str(nb_path))
        assert success is True
        assert "numpy" in imports
        assert len(code_sources) == 1
        assert err is None
        assert lang_label == StatusLabel.PYTHON

    def test_path_a_missing_file_returns_error(self) -> None:
        # Unpack 7-tuple
        success, imports, submodules, code_sources, err, lang_label, guarded = ne.extract_from_file("does_not_exist.ipynb")
        assert success is False
        assert err is not None and len(err) > 0
        assert lang_label == StatusLabel.UNKNOWN

    def test_path_a_corrupted_json_returns_error(self, tmp_path: Path) -> None:
        bad_path: Path = tmp_path / "bad.ipynb"
        bad_path.write_text("{not valid json", encoding="utf-8")

        # Unpack 7-tuple
        success, imports, submodules, code_sources, err, lang_label, guarded = ne.extract_from_file(str(bad_path))
        assert success is False
        assert lang_label == StatusLabel.CORRUPTED

    def test_path_b_reads_live_session_history(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_main = types.ModuleType("__main__")
        fake_main.In = ["", "import requests", "import pandas as pd"]
        monkeypatch.setitem(sys.modules, "__main__", fake_main)

        # Unpack 4-tuple
        imports, submodules, code_sources, guarded = ne.extract_from_active_session()
        assert "requests" in imports
        assert "pandas" in imports

    def test_path_b_empty_history_returns_empty_manifest(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_main = types.ModuleType("__main__")
        fake_main.In = []
        monkeypatch.setitem(sys.modules, "__main__", fake_main)

        # Unpack 4-tuple
        imports, submodules, code_sources, guarded = ne.extract_from_active_session()
        assert imports == set()
        assert code_sources == []


# =====================================================================
# 5. HARDWARE ACCELERATION INSPECTION
# =====================================================================

def _install_fake_module(monkeypatch: pytest.MonkeyPatch, name: str, module: Any) -> None:
    monkeypatch.setitem(sys.modules, name, module)


class TestGpuInspection:
    """Tests runtime hardware inspection across PyTorch, TensorFlow, and JAX."""

    def test_no_frameworks_imported_skips_check(self) -> None:
        # Act
        result: Optional[GpuInfo] = ne.inspect_gpu_environment({"pandas", "requests"})

        # Assert
        assert result is None

    def test_torch_cuda_active(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda i: "NVIDIA GeForce RTX 3090",
        )
        fake_torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
        _install_fake_module(monkeypatch, "torch", fake_torch)

        # Act
        result: Optional[GpuInfo] = ne.inspect_gpu_environment({"torch"})

        # Assert
        assert result is not None
        assert result["has_gpu"] is True
        assert result["active_framework"] == "PyTorch"
        assert "RTX 3090" in result["device_name"]

    def test_torch_mps_active(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        fake_torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True))
        _install_fake_module(monkeypatch, "torch", fake_torch)

        # Act
        result: Optional[GpuInfo] = ne.inspect_gpu_environment({"torch"})

        # Assert
        assert result is not None
        assert result["has_gpu"] is True
        assert "Metal" in result["device_name"]

    def test_torch_imported_no_gpu_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        fake_torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
        _install_fake_module(monkeypatch, "torch", fake_torch)

        # Act
        result: Optional[GpuInfo] = ne.inspect_gpu_environment({"torch"})

        # Assert
        assert result is not None
        assert result["has_gpu"] is False
        assert result["frameworks"] == ["torch"]

    def test_tensorflow_gpu_active(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange
        fake_tf = types.ModuleType("tensorflow")
        fake_gpu_device = object()
        fake_tf.config = types.SimpleNamespace(
            list_physical_devices=lambda kind: [fake_gpu_device] if kind == "GPU" else [],
            experimental=types.SimpleNamespace(
                get_device_details=lambda d: {"device_name": "Tesla T4"}
            ),
        )
        _install_fake_module(monkeypatch, "tensorflow", fake_tf)

        # Act
        result: Optional[GpuInfo] = ne.inspect_gpu_environment({"tensorflow"})

        # Assert
        assert result is not None
        assert result["has_gpu"] is True
        assert result["active_framework"] == "TensorFlow"
        assert "Tesla T4" in result["device_name"]

    def test_jax_accelerator_active(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange
        fake_jax = types.ModuleType("jax")
        fake_device = types.SimpleNamespace(platform="gpu", device_kind="A100")
        fake_jax.devices = lambda: [fake_device]
        _install_fake_module(monkeypatch, "jax", fake_jax)

        # Act
        result: Optional[GpuInfo] = ne.inspect_gpu_environment({"jax"})

        # Assert
        assert result is not None
        assert result["has_gpu"] is True
        assert result["active_framework"] == "JAX"

    def test_framework_not_installed_falls_back_gracefully(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange
        monkeypatch.setitem(sys.modules, "torch", None)

        # Act
        result: Optional[GpuInfo] = ne.inspect_gpu_environment({"torch"})

        # Assert
        assert result is not None
        assert result["has_gpu"] is False


# =====================================================================
# 6. REQUIREMENT CORRELATION & BLUEPRINT GENERATION
# =====================================================================

class TestPackageRequirements:
    """Tests correlation between pinned package specs and extra index URLs."""

    def test_local_tag_without_harvested_url_warns(self) -> None:
        # Act
        manifest, tagged, warnings = ne.process_package_requirements(["torch==2.3.1+cu121"], set())

        # Assert
        assert "torch==2.3.1+cu121" in warnings
        assert tagged == [("torch==2.3.1+cu121", [])]

    def test_local_tag_with_harvested_url_no_warning(self) -> None:
        # Act
        manifest, tagged, warnings = ne.process_package_requirements(
            ["torch==2.3.1+cu121"], {"https://download.pytorch.org/whl/cu121"}
        )

        # Assert
        assert warnings == []
        assert "--extra-index-url https://download.pytorch.org/whl/cu121" in manifest
        assert tagged[0][0] == "torch==2.3.1+cu121"
        assert "https://download.pytorch.org/whl/cu121" in tagged[0][1]

    def test_uninstalled_top_level_import_placeholder(self) -> None:
        # Arrange
        pinned_manifest: List[str] = ["# some_unknown_pkg (imported as 'some_unknown_pkg', not currently found in active env)"]

        # Act
        manifest, tagged, warnings = ne.process_package_requirements(pinned_manifest, set())

        # Assert
        assert manifest == pinned_manifest
        assert tagged == []
        assert warnings == []


class TestBlueprintGeneration:
    """Tests generation of Cell 1 Markdown and Cell 2 Python execution code."""

    def test_returns_both_sections(self) -> None:
        # Act
        blueprint: BlueprintResult = ne.generate_production_blueprint(["numpy==1.26.0"])

        # Assert
        assert "step1_markdown" in blueprint
        assert "step2_code" in blueprint

    def test_python_version_guard_matches_runtime(self) -> None:
        # Act
        blueprint: BlueprintResult = ne.generate_production_blueprint(["numpy==1.26.0"])
        expected_guard: str = f"REQUIRED_PYTHON = ({sys.version_info.major}, {sys.version_info.minor})"

        # Assert
        assert expected_guard in blueprint["step2_code"]

    def test_gpu_section_included_when_gpu_present(self) -> None:
        # Arrange
        gpu_info: GpuInfo = {
            "has_gpu": True,
            "active_framework": "PyTorch",
            "device_name": "NVIDIA GeForce RTX 3090 (via PyTorch)",
            "frameworks": ["torch"],
        }

        # Act
        blueprint: BlueprintResult = ne.generate_production_blueprint(["torch==2.3.1"], gpu_info=gpu_info)

        # Assert
        assert "RTX 3090" in blueprint["step1_markdown"]

    def test_gpu_section_omitted_when_no_gpu(self) -> None:
        # Act
        blueprint: BlueprintResult = ne.generate_production_blueprint(["numpy==1.26.0"], gpu_info=None)

        # Assert
        assert "Hardware Acceleration" not in blueprint["step1_markdown"]

    def test_full_freeze_appended_after_manifest(self) -> None:
        # Act
        blueprint: BlueprintResult = ne.generate_production_blueprint(
            ["numpy==1.26.0"], full_freeze_lines=["# certifi==2024.2.2"]
        )
        code: str = blueprint["step2_code"]
        manifest_pos: int = code.find("numpy==1.26.0")
        freeze_pos: int = code.find("certifi==2024.2.2")

        # Assert
        assert manifest_pos != -1 and freeze_pos != -1
        assert manifest_pos < freeze_pos


# =====================================================================
# 7. RUNTIME SANDBOX EXECUTION
# =====================================================================

class TestRuntimeExecution:
    """Tests real in-memory execution of generated Cell 2 Python code."""

    def test_generated_step2_code_executes_and_writes_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        manifest: List[str] = ["numpy==1.26.4", "pandas==2.2.1"]
        blueprint: BlueprintResult = ne.generate_production_blueprint(manifest)
        
        monkeypatch.chdir(tmp_path)
        
        import subprocess
        monkeypatch.setattr(
            subprocess, 
            "run", 
            lambda *args, **kwargs: types.SimpleNamespace(returncode=0)
        )
        
        # Act: Compile and execute the generated Step 2 Python code string
        compiled_code = compile(blueprint["step2_code"], "<string>", "exec")
        exec_scope: Dict[str, Any] = {"__builtins__": __builtins__}
        exec(compiled_code, exec_scope)
        
        # Assert: The executed code should have written pinned_requirements.txt to disk
        req_file: Path = tmp_path / "pinned_requirements.txt"
        assert req_file.exists()
        
        content: str = req_file.read_text(encoding="utf-8")
        assert "numpy==1.26.4" in content
        assert "pandas==2.2.1" in content
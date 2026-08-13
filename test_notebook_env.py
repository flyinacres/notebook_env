"""
Tests for notebook_env.py (v28).

Unit and integration test suite exercising AST parsing, index URL harvesting,
guarded import tracking (try/except, if/else), dynamic import parsing,
dynamic package/extras resolution, dual-path ingestion, GPU inspection,
blueprint generation, and runtime sandbox execution.
"""

import json
import sys
import types
import warnings
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
        imports, _, _, _ = ne.extract_imports_from_sources(sources)
        assert "numpy" in imports
        assert "os" in imports
        assert "sys" in imports
        assert "sklearn" in imports

    def test_deep_submodule_import_resolves_top_level(self) -> None:
        sources: List[str] = ["import torch.nn.functional as F"]
        imports, submodules, _, _ = ne.extract_imports_from_sources(sources)
        assert "torch" in imports
        assert "torch.nn.functional" in submodules.get("torch", set())

    def test_magics_and_shell_escapes_stripped(self) -> None:
        sources: List[str] = [
            "%matplotlib inline\n%%writefile foo.py\n!pip install foo\nimport pandas as pd"
        ]
        imports, _, _, _ = ne.extract_imports_from_sources(sources)
        assert "pandas" in imports

    def test_syntax_error_in_one_cell_does_not_block_others(self) -> None:
        sources: List[str] = [
            "import pandas as pd",
            "def foo(",
            "import requests",
        ]
        imports, _, _, _ = ne.extract_imports_from_sources(sources)
        assert "pandas" in imports
        assert "requests" in imports

    def test_empty_and_non_code_sources_return_empty(self) -> None:
        imports, submodules, guarded, dyn_warns = ne.extract_imports_from_sources([])
        assert imports == set()
        assert submodules == {}
        assert guarded == set()
        assert dyn_warns == []

    def test_commented_import_not_extracted(self) -> None:
        sources: List[str] = ["# import tensorflow as tf\nimport json"]
        imports, _, _, _ = ne.extract_imports_from_sources(sources)
        assert "tensorflow" not in imports
        assert "json" in imports

    def test_syntax_warning_suppressed_during_ast_parse(self) -> None:
        sources = [
            "x = '\\w+\\d+'\n"
            "import math"
        ]
        
        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            imports, _, _, _ = ne.extract_imports_from_sources(sources)
            
        syntax_warnings = [w for w in recorded_warnings if issubclass(w.category, SyntaxWarning)]
        assert len(syntax_warnings) == 0
        assert "math" in imports


class TestGuardedImports:
    """Tests detection of guarded imports inside try/except and if/else blocks."""

    def test_try_except_import_marked_as_guarded(self) -> None:
        sources = [
            "import numpy as np\n"
            "try:\n"
            "    import cupy as cp\n"
            "except ImportError:\n"
            "    pass"
        ]
        imports, submodules, guarded_imports, _ = ne.extract_imports_from_sources(sources)
        assert "numpy" in imports
        assert "numpy" not in guarded_imports
        assert "cupy" in guarded_imports

    def test_if_statement_import_marked_as_guarded(self) -> None:
        sources = [
            "import os\n"
            "if sys.platform == 'win32':\n"
            "    import pywin32\n"
        ]
        imports, submodules, guarded_imports, _ = ne.extract_imports_from_sources(sources)
        assert "os" in imports
        assert "os" not in guarded_imports
        assert "pywin32" in guarded_imports

    def test_uninstalled_guarded_import_formatted_as_optional_in_manifest(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        frozen_env = {"numpy": "numpy==1.26.4"}
        submodules = {}
        imports = {"numpy", "cupy"}
        guarded_imports = {"cupy"}

        pinned_entries, notices = ne.build_manifest_entries(
            imports, submodules, frozen_env, guarded_imports=guarded_imports
        )

        cupy_pin = next((p for p in pinned_entries if "cupy" in p), "")
        assert cupy_pin.startswith("#")
        assert "guarded" in cupy_pin.lower() or "optional" in cupy_pin.lower()

    def test_installed_guarded_package_emitted_as_optional_comment(self) -> None:
        """Installed packages inside try/except must NOT become hard pins in the manifest."""
        frozen_env = {"cupy": "cupy==13.0.0", "numpy": "numpy==1.26.4"}
        submodules = {}
        imports = {"numpy", "cupy"}
        guarded_imports = {"cupy"}

        pinned_entries, notices = ne.build_manifest_entries(
            imports, submodules, frozen_env, guarded_imports=guarded_imports
        )

        cupy_pin = next((p for p in pinned_entries if "cupy" in p), "")
        assert cupy_pin.startswith("#")
        assert "cupy==13.0.0" in cupy_pin
        assert "guarded" in cupy_pin.lower() or "optional" in cupy_pin.lower()

    def test_unconditional_import_overrides_guarded_import(self) -> None:
        """If a package is imported unconditionally in one cell and guarded in another, it is mandatory."""
        sources = [
            "import numpy as np\n",
            "if sys.platform == 'win32':\n    import numpy as np2\n"
        ]

        imports, submodules, guarded_imports, _ = ne.extract_imports_from_sources(sources)

        assert "numpy" in imports
        assert "numpy" not in guarded_imports


class TestDynamicImportHandling:
    """Tests importlib.import_module and __import__ parsing behavior."""

    def test_literal_string_dynamic_import_extracted(self) -> None:
        """importlib.import_module('torch') with a string literal is extracted."""
        sources = [
            "import importlib\n"
            "torch = importlib.import_module('torch')\n"
        ]

        imports, submodules, guarded_imports, warnings = ne.extract_imports_from_sources(sources)

        assert "torch" in imports
        assert "importlib" in imports
        assert warnings == []

    def test_variable_dynamic_import_emits_warning(self) -> None:
        """importlib.import_module(var_name) emits a diagnostic warning instead of guessing."""
        sources = [
            "import importlib\n"
            "pkg_name = 'tensorflow'\n"
            "mod = importlib.import_module(pkg_name)\n"
        ]

        imports, submodules, guarded_imports, warnings = ne.extract_imports_from_sources(sources)

        assert "tensorflow" not in imports
        assert any("pkg_name" in w for w in warnings)

    def test_from_importlib_import_module_extracted(self) -> None:
        """from importlib import import_module; import_module('torch') is extracted."""
        sources = [
            "from importlib import import_module\n"
            "torch = import_module('torch')\n"
        ]

        imports, submodules, guarded_imports, warnings = ne.extract_imports_from_sources(sources)

        assert "torch" in imports
        assert "importlib" in imports
        assert warnings == []

    def test_aliased_importlib_module_extracted(self) -> None:
        """import importlib as il; il.import_module('torch') is extracted."""
        sources = [
            "import importlib as il\n"
            "torch = il.import_module('torch')\n"
        ]

        imports, submodules, guarded_imports, warnings = ne.extract_imports_from_sources(sources)

        assert "torch" in imports
        assert "importlib" in imports
        assert warnings == []

    def test_from_importlib_with_variable_emits_warning(self) -> None:
        """from importlib import import_module; import_module(var) emits warning."""
        sources = [
            "from importlib import import_module\n"
            "pkg = 'tensorflow'\n"
            "mod = import_module(pkg)\n"
        ]

        imports, submodules, guarded_imports, warnings = ne.extract_imports_from_sources(sources)

        assert "tensorflow" not in imports
        assert any("pkg" in w for w in warnings)
# =====================================================================
# 2. INDEX URL HARVESTING
# =====================================================================

class TestIndexUrlHarvesting:
    def test_extra_index_url_flag(self) -> None:
        sources: List[str] = ["!pip install torch --extra-index-url https://download.pytorch.org/whl/cu121"]
        urls: Set[str] = ne.harvest_index_urls_from_sources(sources)
        assert "https://download.pytorch.org/whl/cu121" in urls

    def test_short_flag(self) -> None:
        sources: List[str] = ["!pip install -i https://pypi.org/simple somepkg"]
        urls: Set[str] = ne.harvest_index_urls_from_sources(sources)
        assert "https://pypi.org/simple" in urls

    def test_quoted_and_multiple_urls_across_cells(self) -> None:
        sources: List[str] = [
            "!pip install foo --extra-index-url 'https://a.example.com'",
            '!pip install bar --extra-index-url "https://b.example.com"',
        ]
        urls: Set[str] = ne.harvest_index_urls_from_sources(sources)
        assert "https://a.example.com" in urls
        assert "https://b.example.com" in urls

    def test_malformed_flag_no_url_not_captured(self) -> None:
        sources: List[str] = ["!pip install foo --extra-index-url"]
        urls: Set[str] = ne.harvest_index_urls_from_sources(sources)
        assert urls == set()

    def test_commented_out_pip_call_not_harvested(self) -> None:
        sources: List[str] = ["# !pip install torch --extra-index-url https://download.pytorch.org/whl/cu121"]
        urls: Set[str] = ne.harvest_index_urls_from_sources(sources)
        assert urls == set()


# =====================================================================
# 3. DYNAMIC RESOLUTION & PROVIDES-EXTRA PROMOTION
# =====================================================================

class TestDynamicResolution:
    def test_submodule_import_promotes_to_package_extras(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_dist = types.SimpleNamespace(
            metadata=types.SimpleNamespace(get_all=lambda key: ["plot"] if key == "Provides-Extra" else [])
        )
        monkeypatch.setattr(importlib.metadata, "distribution", lambda pkg: fake_dist)
        if hasattr(importlib.metadata, "packages_distributions"):
            monkeypatch.setattr(importlib.metadata, "packages_distributions", lambda: {"umap": ["umap-learn"]})

        frozen_env: Dict[str, str] = {"umap-learn": "umap-learn==0.5.5"}
        submodules_set: Set[str] = {"umap.plot"}

        pin, notice = ne.resolve_pypi_package_and_extras("umap", submodules_set, frozen_env)

        assert pin == "umap-learn[plot]==0.5.5"
        assert notice is not None
        assert "umap-learn[plot]==0.5.5" in notice

    def test_fallback_map_used_when_uninstalled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        if hasattr(importlib.metadata, "packages_distributions"):
            monkeypatch.setattr(importlib.metadata, "packages_distributions", lambda: {})

        frozen_env: Dict[str, str] = {}

        pin, notice = ne.resolve_pypi_package_and_extras("sklearn", set(), frozen_env)

        assert pin.startswith("#")
        assert "scikit-learn" in pin
        assert "sklearn" in pin
        assert notice is None


# =====================================================================
# 4. DUAL-PATH INGESTION ENGINE
# =====================================================================

class TestDualPathIngestion:
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

        success, imports, submodules, code_sources, err, lang_label, guarded, dyn_warns = ne.extract_from_file(str(nb_path))
        assert success is True
        assert "numpy" in imports
        assert len(code_sources) == 1
        assert err is None
        assert lang_label == StatusLabel.PYTHON

    def test_path_a_prints_active_interpreter(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        nb: Dict[str, Any] = {"cells": [{"cell_type": "code", "source": ["import math\n"]}]}
        nb_path: Path = tmp_path / "test_interpreter.ipynb"
        nb_path.write_text(json.dumps(nb), encoding="utf-8")

        monkeypatch.setattr(sys, "argv", ["notebook_env.py", str(nb_path)])
        
        ne.main()
        assert sys.executable in caplog.text

    def test_uninstalled_package_produces_fallback_comment_in_main(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        nb: Dict[str, Any] = {"cells": [{"cell_type": "code", "source": ["import fake_uninstalled_pkg\n"]}]}
        nb_path: Path = tmp_path / "test_uninstalled.ipynb"
        nb_path.write_text(json.dumps(nb), encoding="utf-8")

        monkeypatch.setattr(ne, "get_installed_environment", lambda: ({}, []))
        monkeypatch.setattr(sys, "argv", ["notebook_env.py", str(nb_path)])

        ne.main()
        captured_stdout: str = capsys.readouterr().out

        assert "#" in captured_stdout
        assert "fake_uninstalled_pkg" in captured_stdout

    def test_path_a_missing_file_returns_error(self) -> None:
        success, imports, submodules, code_sources, err, lang_label, guarded, dyn_warns = ne.extract_from_file("does_not_exist.ipynb")
        assert success is False
        assert err is not None and len(err) > 0
        assert lang_label == StatusLabel.UNKNOWN

    def test_path_a_corrupted_json_returns_error(self, tmp_path: Path) -> None:
        bad_path: Path = tmp_path / "bad.ipynb"
        bad_path.write_text("{not valid json", encoding="utf-8")

        success, imports, submodules, code_sources, err, lang_label, guarded, dyn_warns = ne.extract_from_file(str(bad_path))
        assert success is False
        assert lang_label == StatusLabel.CORRUPTED

    def test_path_b_reads_live_session_history(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_main = types.ModuleType("__main__")
        fake_main.In = ["", "import requests", "import pandas as pd"]
        monkeypatch.setitem(sys.modules, "__main__", fake_main)

        imports, submodules, code_sources, guarded, dyn_warns = ne.extract_from_active_session()
        assert "requests" in imports
        assert "pandas" in imports

    def test_path_b_empty_history_returns_empty_manifest(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_main = types.ModuleType("__main__")
        fake_main.In = []
        monkeypatch.setitem(sys.modules, "__main__", fake_main)

        imports, submodules, code_sources, guarded, dyn_warns = ne.extract_from_active_session()
        assert imports == set()
        assert code_sources == []


# =====================================================================
# 5. HARDWARE ACCELERATION INSPECTION
# =====================================================================

def _install_fake_module(monkeypatch: pytest.MonkeyPatch, name: str, module: Any) -> None:
    monkeypatch.setitem(sys.modules, name, module)


class TestGpuInspection:
    def test_no_frameworks_imported_skips_check(self) -> None:
        result: Optional[GpuInfo] = ne.inspect_gpu_environment({"pandas", "requests"})
        assert result is None

    def test_torch_cuda_active(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda i: "NVIDIA GeForce RTX 3090",
        )
        fake_torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
        _install_fake_module(monkeypatch, "torch", fake_torch)

        result: Optional[GpuInfo] = ne.inspect_gpu_environment({"torch"})
        assert result is not None
        assert result.has_gpu is True
        assert result.active_framework == "PyTorch"
        assert "RTX 3090" in result.device_name

    def test_torch_mps_active(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        fake_torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True))
        _install_fake_module(monkeypatch, "torch", fake_torch)

        result: Optional[GpuInfo] = ne.inspect_gpu_environment({"torch"})
        assert result is not None
        assert result.has_gpu is True
        assert "Metal" in result.device_name

    def test_torch_imported_no_gpu_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        fake_torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
        _install_fake_module(monkeypatch, "torch", fake_torch)

        result: Optional[GpuInfo] = ne.inspect_gpu_environment({"torch"})
        assert result is not None
        assert result.has_gpu is False
        assert result.frameworks == ["torch"]

    def test_tensorflow_gpu_active(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_tf = types.ModuleType("tensorflow")
        fake_gpu_device = object()
        fake_tf.config = types.SimpleNamespace(
            list_physical_devices=lambda kind: [fake_gpu_device] if kind == "GPU" else [],
            experimental=types.SimpleNamespace(
                get_device_details=lambda d: {"device_name": "Tesla T4"}
            ),
        )
        _install_fake_module(monkeypatch, "tensorflow", fake_tf)

        result: Optional[GpuInfo] = ne.inspect_gpu_environment({"tensorflow"})
        assert result is not None
        assert result.has_gpu is True
        assert result.active_framework == "TensorFlow"
        assert "Tesla T4" in result.device_name

    def test_jax_accelerator_active(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_jax = types.ModuleType("jax")
        fake_device = types.SimpleNamespace(platform="gpu", device_kind="A100")
        fake_jax.devices = lambda: [fake_device]
        _install_fake_module(monkeypatch, "jax", fake_jax)

        result: Optional[GpuInfo] = ne.inspect_gpu_environment({"jax"})
        assert result is not None
        assert result.has_gpu is True
        assert result.active_framework == "JAX"

    def test_jax_metal_active(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """jax-metal reports platform='METAL' (uppercase), not 'gpu'/'tpu' — see probe_jax_gpu."""
        fake_jax = types.ModuleType("jax")
        fake_device = types.SimpleNamespace(platform="METAL", device_kind="Metal")
        fake_jax.devices = lambda: [fake_device]
        _install_fake_module(monkeypatch, "jax", fake_jax)

        result: Optional[GpuInfo] = ne.inspect_gpu_environment({"jax"})
        assert result is not None
        assert result.has_gpu is True
        assert result.active_framework == "JAX"
        assert "METAL" in result.device_name

    def test_framework_not_installed_falls_back_gracefully(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "torch", None)
        result: Optional[GpuInfo] = ne.inspect_gpu_environment({"torch"})
        assert result is not None
        assert result.has_gpu is False

    def test_fastai_transitive_torch_gpu_detection(self, monkeypatch) -> None:
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda i: "NVIDIA RTX 3090",
        )
        fake_torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
        _install_fake_module(monkeypatch, "torch", fake_torch)

        # Note: imported_packages contains 'fastai' but NOT 'torch' directly
        result: Optional[GpuInfo] = ne.inspect_gpu_environment({"fastai"})
        assert result is not None
        assert result.has_gpu is True
        assert result.active_framework == "PyTorch"
        assert "RTX 3090" in result.device_name


# =====================================================================
# 6. REQUIREMENT CORRELATION & BLUEPRINT GENERATION
# =====================================================================

class TestPackageRequirements:
    def test_local_tag_without_harvested_url_warns(self) -> None:
        manifest, tagged, warnings = ne.process_package_requirements(["torch==2.3.1+cu121"], set())
        assert "torch==2.3.1+cu121" in warnings
        assert tagged == [("torch==2.3.1+cu121", [])]

    def test_local_tag_with_harvested_url_no_warning(self) -> None:
        manifest, tagged, warnings = ne.process_package_requirements(
            ["torch==2.3.1+cu121"], {"https://download.pytorch.org/whl/cu121"}
        )
        assert warnings == []
        assert "--extra-index-url https://download.pytorch.org/whl/cu121" in manifest
        assert tagged[0][0] == "torch==2.3.1+cu121"
        assert "https://download.pytorch.org/whl/cu121" in tagged[0][1]

    def test_uninstalled_top_level_import_placeholder(self) -> None:
        pinned_manifest: List[str] = ["# some_unknown_pkg (imported as 'some_unknown_pkg', not currently found in active env)"]
        manifest, tagged, warnings = ne.process_package_requirements(pinned_manifest, set())
        assert manifest == pinned_manifest
        assert tagged == []
        assert warnings == []


class TestBlueprintGeneration:
    def test_returns_both_sections(self) -> None:
        blueprint: BlueprintResult = ne.generate_production_blueprint(["numpy==1.26.0"])
        assert "step1_markdown" in blueprint
        assert "step2_code" in blueprint

    def test_python_version_guard_matches_runtime(self) -> None:
        blueprint: BlueprintResult = ne.generate_production_blueprint(["numpy==1.26.0"])
        expected_guard: str = f"REQUIRED_PYTHON = ({sys.version_info.major}, {sys.version_info.minor})"
        assert expected_guard in blueprint["step2_code"]

    def test_gpu_section_included_when_gpu_present(self) -> None:
        gpu_info = ne.GpuInfo(
            has_gpu=True,
            active_framework="PyTorch",
            device_name="NVIDIA GeForce RTX 3090 (via PyTorch)",
            frameworks=["torch"],
        )
        blueprint: BlueprintResult = ne.generate_production_blueprint(["torch==2.3.1"], gpu_info=gpu_info)
        assert "RTX 3090" in blueprint["step1_markdown"]

    def test_gpu_section_omitted_when_no_gpu(self) -> None:
        blueprint: BlueprintResult = ne.generate_production_blueprint(["numpy==1.26.0"], gpu_info=None)
        assert "Hardware Acceleration" not in blueprint["step1_markdown"]

    def test_full_freeze_appended_after_manifest(self) -> None:
        blueprint: BlueprintResult = ne.generate_production_blueprint(
            ["numpy==1.26.0"], full_freeze_lines=["# certifi==2024.2.2"]
        )
        code: str = blueprint["step2_code"]
        manifest_pos: int = code.find("numpy==1.26.0")
        freeze_pos: int = code.find("certifi==2024.2.2")
        assert manifest_pos != -1 and freeze_pos != -1
        assert manifest_pos < freeze_pos


# =====================================================================
# 7. RUNTIME SANDBOX EXECUTION
# =====================================================================

class TestRuntimeExecution:
    def test_generated_step2_code_executes_and_writes_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manifest: List[str] = ["numpy==1.26.4", "pandas==2.2.1"]
        blueprint: BlueprintResult = ne.generate_production_blueprint(manifest)
        
        monkeypatch.chdir(tmp_path)
        
        import subprocess
        monkeypatch.setattr(
            subprocess, 
            "run", 
            lambda *args, **kwargs: types.SimpleNamespace(returncode=0)
        )
        
        compiled_code = compile(blueprint["step2_code"], "<string>", "exec")
        exec_scope: Dict[str, Any] = {"__builtins__": __builtins__}
        exec(compiled_code, exec_scope)
        
        req_file: Path = tmp_path / "pinned_requirements.txt"
        assert req_file.exists()
        
        content: str = req_file.read_text(encoding="utf-8")
        assert "numpy==1.26.4" in content
        assert "pandas==2.2.1" in content

class TestCellClassificationAndMagicHarvesting:
    """Tests Phase 2 cell classification and magic/shell command harvesting."""

    def test_bash_cell_bypasses_ast_parse_without_syntax_error(self) -> None:
        """%%bash cells bypass ast.parse so invalid Python syntax doesn't crash or get silently dropped."""
        sources = [
            "%%bash\n"
            "apt-get update && apt-get install -y graphviz\n"
            "pip install gdown\n"
        ]

        harvested_pkgs, base_urls, extra_urls, warnings, notices = ne.harvest_cell_magics_and_commands(sources)

        assert "gdown" in harvested_pkgs
        assert any("apt-get" in n.lower() or "system" in n.lower() for n in notices)

    def test_distinguishes_index_url_from_extra_index_url(self) -> None:
        """Base --index-url and supplemental --extra-index-url are categorized separately."""
        sources = [
            "%pip install torch --index-url https://custom.base.index/simple\n",
            "!pip install torchvision --extra-index-url https://download.pytorch.org/whl/cu121\n"
        ]

        harvested_pkgs, base_urls, extra_urls, warnings, notices = ne.harvest_cell_magics_and_commands(sources)

        assert "https://custom.base.index/simple" in base_urls
        assert "https://download.pytorch.org/whl/cu121" in extra_urls

    def test_conda_install_logs_informational_notice(self) -> None:
        """%conda or !conda installs generate an informational notice rather than pip freeze correlation."""
        sources = ["%conda install -c conda-forge graphviz\n"]

        harvested_pkgs, base_urls, extra_urls, warnings, notices = ne.harvest_cell_magics_and_commands(sources)

        assert any("conda" in n.lower() for n in notices)

    def test_requirements_file_reference_logs_warning(self) -> None:
        """%pip install -r requirements.txt emits a diagnostic warning."""
        sources = ["!pip install -r requirements.txt\n"]

        harvested_pkgs, base_urls, extra_urls, warnings, notices = ne.harvest_cell_magics_and_commands(sources)

        assert any("requirements" in w.lower() for w in warnings)

class TestIntegrationAndFormatting:
    """Tests Phase 3 manifest section ordering, auxiliary package correlation, and index placement."""

    def test_base_index_placed_at_top_of_manifest(self) -> None:
        """--index-url appears at the very top of the generated manifest lines before dependencies."""
        pinned = ["pandas==2.2.0", "torch==2.2.0"]
        harvested_urls = {"https://download.pytorch.org/whl/cu121"}
        base_urls = {"https://custom.pypi.org/simple"}

        manifest_lines, _, _ = ne.process_package_requirements(
            pinned, harvested_urls, base_urls=base_urls
        )

        assert manifest_lines[0] == "--index-url https://custom.pypi.org/simple"
        assert manifest_lines[1] == "--extra-index-url https://download.pytorch.org/whl/cu121"

    def test_auxiliary_tools_rendered_in_separate_commented_block(self) -> None:
        """Unimported auxiliary packages harvested from magics are rendered in a dedicated commented block."""
        imports = {"pandas"}
        harvested_pkgs = {"gdown", "pandas"}  # pandas is already imported, gdown is aux-only
        frozen_env = {"pandas": "pandas==2.2.0", "gdown": "gdown==5.1.0"}

        aux_entries = ne.build_auxiliary_tool_entries(harvested_pkgs, imports, frozen_env)

        assert len(aux_entries) == 2
        assert aux_entries[0] == "\n# --- AUXILIARY TOOL INSTALLS (harvested from cell magics) ---"
        assert "# gdown==5.1.0" in aux_entries[1]
        assert "installed via cell command" in aux_entries[1]


    def test_uninstalled_auxiliary_tools_rendered_as_unpinned_comment(self) -> None:
        """Auxiliary tools not found in the active environment render as unpinned commented entries."""
        imports = set()
        harvested_pkgs = {"awscli"}
        frozen_env = {}

        aux_entries = ne.build_auxiliary_tool_entries(harvested_pkgs, imports, frozen_env)

        assert len(aux_entries) == 2
        assert "# awscli  (installed via cell command; not found in active env)" in aux_entries[1]
    def test_writefile_script_dependencies_rendered_in_separate_section(self) -> None:
        """Dependencies imported exclusively inside %%writefile cells render in a dedicated block."""
        primary_imports = {"pandas"}
        writefile_imports = {"requests", "pandas"}  # pandas is in primary, requests is script-only
        frozen_env = {"pandas": "pandas==2.30.0", "requests": "requests==2.31.0"}

        entries = ne.build_writefile_tool_entries(writefile_imports, primary_imports, frozen_env)

        assert len(entries) == 2
        assert entries[0] == "\n# --- WRITEFILE SCRIPT DEPENDENCIES ---"
        assert "# requests==2.31.0" in entries[1]
        assert "imported inside script generated via %%writefile" in entries[1]

def test_discover_local_repo_modules_top_level(tmp_path):
    """Verify discover_local_repo_modules recognizes top-level modules and package directories."""
    src_dir = tmp_path / "src" / "utils"
    src_dir.mkdir(parents=True)
    (src_dir / "helpers.py").write_text("# helper module", encoding="utf-8")
    (tmp_path / "root_script.py").write_text("# root script", encoding="utf-8")

    discovered = ne.discover_local_repo_modules(str(tmp_path))

    assert "src" in discovered
    assert "root_script" in discovered
    assert "helpers" not in discovered

def test_production_blueprint_failure_message_dynamic():
    """Verify Cell 2 failure advice adaptively includes user troubleshooting steps and HELP_URL link."""
    # Standard manifest without local tags
    std_blueprint = ne.generate_production_blueprint(["pandas==2.1.0", "numpy==1.25.0"])
    assert "Internet Access" in std_blueprint["step2_code"]
    assert ne.HELP_URL in std_blueprint["step2_code"]

    # Manifest containing local tag build
    tagged_blueprint = ne.generate_production_blueprint(["torch==2.1.0+cu121"])
    assert "Troubleshooting Steps:" in tagged_blueprint["step2_code"]
    assert ne.HELP_URL in tagged_blueprint["step2_code"]


class TestMemoizeForRun:
    """
    Tests for the _memoize_for_run decorator and its use on
    get_notebook_local_modules / build_manifest_entries.

    Covers the three properties that matter for a shared, mutable-return-type
    cache: (1) repeated identical calls actually skip recomputation,
    (2) distinct inputs are never conflated into the same cache entry, and
    (3) callers can't corrupt the cache by mutating a returned value.
    """

    def test_local_modules_dedupes_identical_calls(self, tmp_path, monkeypatch):
        nb_path = tmp_path / "nb.ipynb"
        nb_path.touch()

        call_count = {"n": 0}
        real_discover = ne.discover_local_repo_modules

        def counting_discover(*args, **kwargs):
            call_count["n"] += 1
            return real_discover(*args, **kwargs)

        monkeypatch.setattr(ne, "discover_local_repo_modules", counting_discover)
        ne.get_notebook_local_modules.cache_clear()

        r1 = ne.get_notebook_local_modules(nb_path, str(tmp_path))
        calls_after_first = call_count["n"]
        r2 = ne.get_notebook_local_modules(nb_path, str(tmp_path))

        assert r1 == r2
        assert call_count["n"] == calls_after_first, "second identical call should not re-scan the filesystem"

    def test_local_modules_different_notebooks_not_conflated(self, tmp_path):
        """Distinct (path, root_dir) inputs must never share a cache entry, even
        under the id()-keying scheme — each notebook has its own Path object."""
        dir_a, dir_b = tmp_path / "a", tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        (dir_a / "helper_a.py").write_text("# a", encoding="utf-8")
        (dir_b / "helper_b.py").write_text("# b", encoding="utf-8")

        ne.get_notebook_local_modules.cache_clear()
        result_a = ne.get_notebook_local_modules(dir_a / "nb.ipynb", str(dir_a))
        result_b = ne.get_notebook_local_modules(dir_b / "nb.ipynb", str(dir_b))

        assert "helper_a" in result_a and "helper_a" not in result_b
        assert "helper_b" in result_b and "helper_b" not in result_a

    def test_local_modules_returns_defensive_copy(self, tmp_path):
        nb_path = tmp_path / "nb.ipynb"
        nb_path.touch()
        ne.get_notebook_local_modules.cache_clear()

        r1 = ne.get_notebook_local_modules(nb_path, str(tmp_path))
        r1.add("INJECTED_BY_TEST")
        r2 = ne.get_notebook_local_modules(nb_path, str(tmp_path))

        assert "INJECTED_BY_TEST" not in r2, "mutating one caller's result must not corrupt the cached value"

    def test_local_modules_cache_clear_forces_recompute(self, tmp_path, monkeypatch):
        nb_path = tmp_path / "nb.ipynb"
        nb_path.touch()

        call_count = {"n": 0}
        real_discover = ne.discover_local_repo_modules

        def counting_discover(*args, **kwargs):
            call_count["n"] += 1
            return real_discover(*args, **kwargs)

        monkeypatch.setattr(ne, "discover_local_repo_modules", counting_discover)
        ne.get_notebook_local_modules.cache_clear()

        ne.get_notebook_local_modules(nb_path, str(tmp_path))
        calls_before_clear = call_count["n"]
        ne.get_notebook_local_modules.cache_clear()
        ne.get_notebook_local_modules(nb_path, str(tmp_path))

        assert call_count["n"] == calls_before_clear * 2, "cache_clear() must force a real recompute, not return stale data"

    def test_build_manifest_entries_dedupes_identical_calls(self, monkeypatch):
        call_count = {"n": 0}
        real_resolve = ne.resolve_pypi_package_and_extras

        def counting_resolve(*args, **kwargs):
            call_count["n"] += 1
            return real_resolve(*args, **kwargs)

        monkeypatch.setattr(ne, "resolve_pypi_package_and_extras", counting_resolve)
        ne.build_manifest_entries.cache_clear()

        imports = {"pandas", "numpy"}
        # Same dict object passed to both calls, matching every real call site
        # (analyze_batch_repository / generate_universal_manifest / apply_output_to_notebook
        # all read the same res.submodules attribute repeatedly, never rebuild it) —
        # see build_manifest_entries's memoization docstring: dict args are keyed
        # by id(), not content, so two *different* dict objects (even empty,
        # equal ones) would correctly miss the cache. See the dedicated test
        # below for that case.
        submodules: Dict[str, Set[str]] = {}
        frozen_env = {"pandas": "pandas==2.2.0", "numpy": "numpy==1.26.0"}

        r1 = ne.build_manifest_entries(imports, submodules, frozen_env)
        calls_after_first = call_count["n"]
        r2 = ne.build_manifest_entries(imports, submodules, frozen_env)

        assert r1 == r2
        assert call_count["n"] == calls_after_first, "second identical call (same object refs) should not re-resolve every import"

    def test_build_manifest_entries_distinct_equal_dicts_not_treated_as_cache_hit(self, monkeypatch):
        """
        Documents an intentional limitation, not a bug: dict arguments are keyed
        by id() rather than content (see _memoize_for_run docstring — re-hashing
        a potentially large frozen_env/pkg_dist_map on every call would cost more
        than caching saves). So two distinct dict objects with equal content are
        correctly treated as a cache MISS, not a false-positive hit. This is the
        safe direction to fail in: the worst case is a missed optimization, never
        a wrong answer served from an unrelated dict.
        """
        call_count = {"n": 0}
        real_resolve = ne.resolve_pypi_package_and_extras

        def counting_resolve(*args, **kwargs):
            call_count["n"] += 1
            return real_resolve(*args, **kwargs)

        monkeypatch.setattr(ne, "resolve_pypi_package_and_extras", counting_resolve)
        ne.build_manifest_entries.cache_clear()

        imports = {"pandas"}
        frozen_env = {"pandas": "pandas==2.2.0"}

        r1 = ne.build_manifest_entries(imports, {}, frozen_env)
        calls_after_first = call_count["n"]
        r2 = ne.build_manifest_entries(imports, {}, frozen_env)  # a *different* {} object, same content

        assert r1 == r2, "results must still be correct even without a cache hit"
        assert call_count["n"] == calls_after_first * 2, "distinct dict objects must not be conflated into one cache entry"

    def test_build_manifest_entries_different_imports_not_conflated(self):
        ne.build_manifest_entries.cache_clear()
        frozen_env = {"pandas": "pandas==2.2.0", "numpy": "numpy==1.26.0"}

        entries_pandas, _ = ne.build_manifest_entries({"pandas"}, {}, frozen_env)
        entries_numpy, _ = ne.build_manifest_entries({"numpy"}, {}, frozen_env)

        assert any("pandas" in line for line in entries_pandas)
        assert not any("pandas" in line for line in entries_numpy)
        assert any("numpy" in line for line in entries_numpy)
        assert not any("numpy" in line for line in entries_pandas)

    def test_build_manifest_entries_different_frozen_env_not_conflated(self):
        """Two distinct frozen_env dict objects (even if built independently)
        must resolve independently — id()-keying must not accidentally treat
        an unrelated dict as a cache hit."""
        ne.build_manifest_entries.cache_clear()
        imports = {"pandas"}

        entries_v1, _ = ne.build_manifest_entries(imports, {}, {"pandas": "pandas==2.2.0"})
        entries_v2, _ = ne.build_manifest_entries(imports, {}, {"pandas": "pandas==1.5.0"})

        assert any("2.2.0" in line for line in entries_v1)
        assert any("1.5.0" in line for line in entries_v2)

    def test_build_manifest_entries_returns_defensive_copy(self):
        ne.build_manifest_entries.cache_clear()
        imports = {"pandas"}
        submodules: Dict[str, Set[str]] = {}  # same object both calls — see dedup test above for why
        frozen_env = {"pandas": "pandas==2.2.0"}

        entries1, notes1 = ne.build_manifest_entries(imports, submodules, frozen_env)
        entries1.append("INJECTED_BY_TEST")
        notes1.append("INJECTED_NOTE")
        entries2, notes2 = ne.build_manifest_entries(imports, submodules, frozen_env)

        assert "INJECTED_BY_TEST" not in entries2
        assert "INJECTED_NOTE" not in notes2

    def test_main_clears_memoization_caches_before_anything_else(self, monkeypatch):
        """main() is the single documented entrypoint for both CLI and live-kernel
        usage (`import notebook_env as ne; ne.main()`). It must clear both memoized
        caches unconditionally, before argument parsing even happens, so a
        long-lived kernel session never returns stale results after the user
        edits files on disk between calls to ne.main()."""
        cleared = {"local_modules": False, "manifest": False}
        monkeypatch.setattr(ne.get_notebook_local_modules, "cache_clear", lambda: cleared.__setitem__("local_modules", True))
        monkeypatch.setattr(ne.build_manifest_entries, "cache_clear", lambda: cleared.__setitem__("manifest", True))

        # --output with no notebook/--batch target hits main()'s validation
        # sys.exit(1) — a real, guaranteed-early exit path that fires *after*
        # the cache_clear() calls at the top of main() but *before* the
        # expensive get_installed_environment() subprocess call, so this
        # verifies clearing happens unconditionally without needing to run
        # the full pipeline. (main() uses parse_known_args(), which silently
        # ignores unrecognized flags rather than erroring, so an invalid-flag
        # approach wouldn't reliably exit here.)
        monkeypatch.setattr(sys, "argv", ["notebook_env.py", "--output"])

        with pytest.raises(SystemExit):
            ne.main()

        assert cleared["local_modules"] is True
        assert cleared["manifest"] is True
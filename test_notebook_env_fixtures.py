"""
Fixture-based end-to-end test using kitchen_sink.ipynb.
Refactored to test structural requirements rather than hardcoded log phrasing.
"""

import json
import sys
import types
import importlib.metadata
from pathlib import Path

import pytest

import notebook_env as ne


FIXTURE_DIR = Path(__file__).parent / "fixtures"
KITCHEN_SINK_PATH = FIXTURE_DIR / "kitchen_sink.ipynb"


@pytest.fixture
def kitchen_sink_notebook():
    if not KITCHEN_SINK_PATH.exists():
        pytest.fail(f"Fixture notebook not found at {KITCHEN_SINK_PATH}.")
    return KITCHEN_SINK_PATH


@pytest.fixture
def mock_environment(monkeypatch):
    frozen_env = {
        "numpy": "numpy==1.26.4",
        "opencv-python": "opencv-python==4.9.0.80",
        "scikit-learn": "scikit-learn==1.4.2",
        "pillow": "pillow==10.3.0",
        "beautifulsoup4": "beautifulsoup4==4.12.3",
        "torch": "torch==2.3.1+cu121",
    }
    raw_freeze = list(frozen_env.values())
    monkeypatch.setattr(ne, "get_installed_environment", lambda: (frozen_env, raw_freeze))
    monkeypatch.setattr(ne, "resolve_opencv_variant", lambda submodules=None: "opencv-python")

    fake_packages_distributions = {
        "numpy": ["numpy"],
        "sklearn": ["scikit-learn"],
        "PIL": ["pillow"],
        "bs4": ["beautifulsoup4"],
        "yaml": ["PyYAML"],
    }
    monkeypatch.setattr(
        importlib.metadata, "packages_distributions",
        lambda: fake_packages_distributions
    )

    def fake_distribution(pkg_name):
        raise importlib.metadata.PackageNotFoundError(pkg_name)
    monkeypatch.setattr(importlib.metadata, "distribution", fake_distribution)

    return frozen_env


class TestKitchenSinkNotebook:
    def test_extraction_only(self, kitchen_sink_notebook):
        """Confirms raw AST extraction, independent of environment correlation."""
        success, imports, submodules, code_sources, error_msg, lang_label, guarded, dyn_warns = ne.extract_from_file(str(kitchen_sink_notebook))
        assert success is True

        for expected in ("numpy", "cv2", "sklearn", "yaml", "PIL", "bs4", "torch", "cupy", "umap", "this_package_does_not_exist_xyz"):
            assert expected in imports, f"expected '{expected}' to be extracted"

        assert list(imports).count("numpy") == 1
        assert "nonexistent_fake_package" not in imports
        assert "fake_package_in_a_string" not in imports
        assert "importlib" in imports

        assert "sklearn.ensemble" in submodules.get("sklearn", set())
        assert "torch.nn" in submodules.get("torch", set())
        assert "umap.plot" in submodules.get("umap", set())

        # Verify try/except guarded imports extracted in kitchen_sink.ipynb
        assert "cupy" in guarded
        assert "this_package_does_not_exist_xyz" in guarded

    def test_stdlib_correctly_filtered(self, kitchen_sink_notebook):
        success, imports, submodules, code_sources, error_msg, lang_label, guarded, dyn_warns = ne.extract_from_file(str(kitchen_sink_notebook))
        non_stdlib = {i for i in imports if i not in ne.STD_LIB}

        for stdlib_name in ("os", "collections", "xml", "json", "re", "itertools", "math", "importlib"):
            assert stdlib_name not in non_stdlib, f"'{stdlib_name}' should have been filtered as stdlib"

    def test_full_pipeline_manifest_content(self, kitchen_sink_notebook, mock_environment, monkeypatch, capsys):
        import subprocess
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=0))
        monkeypatch.setattr(sys, "argv", ["notebook_env.py", str(kitchen_sink_notebook)])

        ne.main()
        out = capsys.readouterr().out

        assert "pillow" in out and "10.3.0" in out
        assert "beautifulsoup4" in out and "4.12.3" in out
        assert "opencv-python" in out and "4.9.0.80" in out
        assert "numpy" in out and "1.26.4" in out
        assert "scikit-learn" in out and "1.4.2" in out
        assert "torch" in out and "2.3.1+cu121" in out

        for uninstalled_pkg in ("cupy", "this_package_does_not_exist_xyz", "umap", "PyYAML"):
            assert uninstalled_pkg in out

        for absent in ("collections==", "xml==", "os==", "nonexistent_fake_package", "fake_package_in_a_string"):
            assert absent not in out

        assert "https://download.pytorch.org/whl/cu121" in out

    def test_relative_import_is_silently_invisible(self, kitchen_sink_notebook):
        success, imports, submodules, code_sources, error_msg, lang_label, guarded, dyn_warns = ne.extract_from_file(str(kitchen_sink_notebook))
        assert success is True
        assert "helper_module" not in imports
        assert not any("helper" in name for name in imports)

    def test_umap_extras_promotion_now_works(self, kitchen_sink_notebook, monkeypatch, capsys, caplog):
        import subprocess

        mock_frozen_env = {
            "numpy": "numpy==1.26.4",
            "opencv-python": "opencv-python==4.9.0.80",
            "scikit-learn": "scikit-learn==1.4.2",
            "pillow": "pillow==10.3.0",
            "beautifulsoup4": "beautifulsoup4==4.12.3",
            "torch": "torch==2.3.1+cu121",
            "umap-learn": "umap-learn==0.5.5",
        }
        mock_raw_freeze = list(mock_frozen_env.values())
        monkeypatch.setattr(ne, "get_installed_environment", lambda: (mock_frozen_env, mock_raw_freeze))
        monkeypatch.setattr(ne, "resolve_opencv_variant", lambda submodules=None: "opencv-python")
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=0))
        monkeypatch.setattr(sys, "argv", ["notebook_env.py", str(kitchen_sink_notebook)])

        fake_packages_distributions = {
            "numpy": ["numpy"],
            "sklearn": ["scikit-learn"],
            "PIL": ["pillow"],
            "bs4": ["beautifulsoup4"],
            "yaml": ["PyYAML"],
            "umap": ["umap-learn"],
        }
        monkeypatch.setattr(importlib.metadata, "packages_distributions", lambda: fake_packages_distributions)

        def fake_distribution(pkg_name):
            if pkg_name == "umap-learn":
                return types.SimpleNamespace(
                    metadata=types.SimpleNamespace(
                        get_all=lambda key: ["plot"] if key == "Provides-Extra" else []
                    )
                )
            raise importlib.metadata.PackageNotFoundError(pkg_name)
        monkeypatch.setattr(importlib.metadata, "distribution", fake_distribution)

        ne.main()
        out = capsys.readouterr().out

        assert "umap-learn" in out
        assert "plot" in out
        assert "0.5.5" in out
        assert "umap.plot" in caplog.text

class TestPypiMapTranslations:
    """Tests resolution of special case package translations."""

    def test_import_to_pypi_translation_misses(self) -> None:
        """dotenv resolves to python-dotenv and mpl_toolkits resolves to matplotlib."""
        frozen_env = {}
        
        pin_dotenv, _ = ne.resolve_pypi_package_and_extras("dotenv", set(), frozen_env)
        assert "python-dotenv" in pin_dotenv.specifier

        pin_mpl, _ = ne.resolve_pypi_package_and_extras("mpl_toolkits", set(), frozen_env)
        assert "matplotlib" in pin_mpl.specifier
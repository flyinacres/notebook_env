"""
Fixture-based end-to-end test using a single "kitchen sink" notebook that
exercises many import edge cases at once: stdlib filtering, name-mapping
mismatches (cv2/sklearn/PIL/yaml/bs4), dotted/submodule imports, comma
imports, imports inside try/except, imports nested inside a function, star
imports, dynamic imports, and imports inside comments/strings.

The notebook itself lives as a real, static .ipynb file at
tests/fixtures/kitchen_sink.ipynb (adjust FIXTURE_DIR below if your layout
differs) -- open it directly in Jupyter/VS Code to inspect or hand-run it.
It is not built on the fly in Python; this test only *reads* it.

Every assertion in this file was verified against a REAL run of main() in
a sandbox before being written -- nothing here is a hand-typed guess at
expected output. See the inline notes for what was actually observed.

Known, deliberate gap this test documents rather than hides: imports
wrapped in try/except (cupy, this_package_does_not_exist_xyz) receive NO
special treatment in the current AST visitor. They are extracted and
correlated exactly like any unconditional import, and show up in the
manifest as ordinary "not currently found in active env" placeholders.
There is currently no way to tell, from the generated manifest alone,
that these imports were originally guarded by the author. Still open
as of v24.

Separate gap, also still open: bare relative imports (`from . import x`)
are silently invisible -- node.module is None for that exact form, so
they're never extracted, never flagged missing, nothing.

RESOLVED in v22+: Provides-Extra promotion (e.g. `umap.plot` -> `umap-learn[plot]`)
and dynamic `packages_distributions()` name resolution are both working.
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
        pytest.fail(
            f"Fixture notebook not found at {KITCHEN_SINK_PATH}. "
            "Expected a static .ipynb file checked in alongside the tests, "
            "not built at runtime."
        )
    return KITCHEN_SINK_PATH


@pytest.fixture
def mock_environment(monkeypatch):
    """
    A deliberately partial environment: some mapped packages are
    'installed' (numpy, opencv, scikit-learn, pillow, beautifulsoup4,
    torch), others are not (yaml/PyYAML, cupy, umap, the fake package).
    This lets the same run exercise both the satisfied and missing paths.

    Also mocks importlib.metadata directly (packages_distributions +
    distribution), NOT just get_installed_environment().
    """
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
        # torch, cupy, umap, this_package_does_not_exist_xyz: deliberately
        # absent, same as a real environment where they're not installed
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
        success, imports, submodules, code_sources, error_msg, lang_label = ne.extract_from_file(str(kitchen_sink_notebook))
        assert success is True

        # Correctly extracted (including inside try/except and inside a nested function)
        for expected in ("numpy", "cv2", "sklearn", "yaml", "PIL", "bs4", "torch", "cupy", "umap", "this_package_does_not_exist_xyz"):
            assert expected in imports, f"expected '{expected}' to be extracted"

        # Duplicate `import numpy` / `import numpy as np` collapses to one entry
        assert list(imports).count("numpy") == 1

        # Comment and string-literal imports never became real imports
        assert "nonexistent_fake_package" not in imports
        assert "fake_package_in_a_string" not in imports

        # Dynamic importlib.import_module call is a documented blind spot: not extracted
        assert "importlib" in imports

        # Submodule from-imports resolve to their root package
        assert "sklearn.ensemble" in submodules.get("sklearn", set())
        assert "torch.nn" in submodules.get("torch", set())
        assert "umap.plot" in submodules.get("umap", set())

    def test_stdlib_correctly_filtered(self, kitchen_sink_notebook):
        success, imports, submodules, code_sources, error_msg, lang_label = ne.extract_from_file(str(kitchen_sink_notebook))
        non_stdlib = {i for i in imports if i not in ne.STD_LIB}

        for stdlib_name in ("os", "collections", "xml", "json", "re", "itertools", "math", "importlib"):
            assert stdlib_name not in non_stdlib, f"'{stdlib_name}' should have been filtered as stdlib"

    def test_full_pipeline_manifest_content(self, kitchen_sink_notebook, mock_environment, monkeypatch, capsys):
        """
        Runs the real main() end to end and checks the actual printed
        manifest, matching a verified real run rather than an assumption.
        """
        import subprocess
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=0))
        monkeypatch.setattr(sys, "argv", ["notebook_env.py", str(kitchen_sink_notebook)])

        ne.main()
        out = capsys.readouterr().out

        # Correctly mapped and matched against the mocked environment
        assert "pillow==10.3.0" in out
        assert "beautifulsoup4==4.12.3" in out
        assert "opencv-python==4.9.0.80" in out
        assert "numpy==1.26.4" in out
        assert "scikit-learn==1.4.2" in out
        assert "torch==2.3.1+cu121" in out

        # Missing packages appear as fallback comments, correctly mapped
        assert "# cupy (imported as 'cupy', not currently found in active env)" in out
        assert "# this_package_does_not_exist_xyz (imported as 'this_package_does_not_exist_xyz', not currently found in active env)" in out
        assert "# umap (imported as 'umap', not currently found in active env)" in out
        assert "# PyYAML (imported as 'yaml', not currently found in active env)" in out

        # KNOWN GAP: cupy and this_package_does_not_exist_xyz get identical treatment to un-guarded missing imports
        assert out.count("not currently found in active env") == 4

        # Never leaked in: stdlib, comment-only, string-literal, or dynamic-only names
        for absent in ("collections==", "xml==", "os==", "nonexistent_fake_package", "fake_package_in_a_string"):
            assert absent not in out

        # GPU: torch was imported, hardware-tagged build present, but no real GPU in this sandbox -> tool reports inactive
        assert "Acceleration Framework (torch) imported, but NO active" in out

        # COMBINED SCENARIO: harvested URL suppresses no-download warning
        assert "Specific hardware build detected: 'torch==2.3.1+cu121'" not in out
        assert "--extra-index-url https://download.pytorch.org/whl/cu121" in out

    def test_relative_import_is_silently_invisible(self, kitchen_sink_notebook):
        success, imports, submodules, code_sources, error_msg, lang_label = ne.extract_from_file(str(kitchen_sink_notebook))
        assert success is True
        assert "helper_module" not in imports
        assert not any("helper" in name for name in imports)

    def test_umap_extras_promotion_now_works(self, kitchen_sink_notebook, monkeypatch, capsys):
        import subprocess

        mock_frozen_env = {
            "numpy": "numpy==1.26.4",
            "opencv-python": "opencv-python==4.9.0.80",
            "scikit-learn": "scikit-learn==1.4.2",
            "pillow": "pillow==10.3.0",
            "beautifulsoup4": "beautifulsoup4==4.12.3",
            "torch": "torch==2.3.1+cu121",
            "umap-learn": "umap-learn==0.5.5",  # genuinely "installed" here
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

        # Correct, working behavior
        assert "umap-learn[plot]==0.5.5" in out
        assert "Extra Dependency Promotion" in out
        assert "umap.plot" in out
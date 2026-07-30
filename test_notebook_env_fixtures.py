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
that these imports were originally guarded by the author. If that
distinction becomes important later, it needs the try/except-awareness
that existed in an earlier version of this tool and was not carried
forward into v21.

Separate known gap, not covered by this fixture: the earlier v16 design
had Provides-Extra handling to promote e.g. `umap.plot` to `umap-learn[plot]`.
That logic is not present in v21 -- `umap.plot` currently just resolves to
a plain `umap` import with no extras awareness. Flagged here rather than
silently worked around.
"""

import json
import sys
import types
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
    return frozen_env


class TestKitchenSinkNotebook:
    def test_extraction_only(self, kitchen_sink_notebook):
        """Confirms raw AST extraction, independent of environment correlation."""
        success, imports, submodules, code_sources, error_msg = ne.extract_from_file(str(kitchen_sink_notebook))
        assert success is True

        # Correctly extracted (including inside try/except and inside a nested function)
        for expected in ("numpy", "cv2", "sklearn", "yaml", "PIL", "bs4", "torch", "cupy", "umap", "this_package_does_not_exist_xyz"):
            assert expected in imports, f"expected '{expected}' to be extracted"

        # Duplicate `import numpy` / `import numpy as np` collapses to one entry
        assert list(imports).count("numpy") == 1  # sets can't actually duplicate; this documents intent

        # Comment and string-literal imports never became real imports
        assert "nonexistent_fake_package" not in imports
        assert "fake_package_in_a_string" not in imports

        # Dynamic importlib.import_module call is a documented blind spot: not extracted
        # ('importlib' itself IS extracted, since it's a literal `import importlib` statement)
        assert "importlib" in imports

        # Submodule from-imports resolve to their root package
        assert "sklearn.ensemble" in submodules.get("sklearn", set())
        assert "torch.nn" in submodules.get("torch", set())
        assert "umap.plot" in submodules.get("umap", set())

    def test_stdlib_correctly_filtered(self, kitchen_sink_notebook):
        success, imports, submodules, code_sources, error_msg = ne.extract_from_file(str(kitchen_sink_notebook))
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

        # KNOWN GAP, asserted explicitly so a future fix to try/except-awareness
        # will surface here as an intentional test update, not a silent behavior change:
        # cupy and this_package_does_not_exist_xyz get IDENTICAL treatment to a
        # top-level unconditional missing import. Nothing marks them as "was optional".
        assert out.count("not currently found in active env") == 4

        # Never leaked in: stdlib, comment-only, string-literal, or dynamic-only names
        for absent in ("collections==", "xml==", "os==", "nonexistent_fake_package", "fake_package_in_a_string"):
            assert absent not in out

        # GPU: torch was imported, hardware-tagged build present, but no real GPU
        # in this sandbox -> tool correctly reports "imported but not active"
        assert "Acceleration Framework (torch) imported, but NO active GPU/TPU accelerator was found" in out
        assert "Specific hardware build detected: 'torch==2.3.1+cu121'" in out
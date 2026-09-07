"""
Phase 5b: Structural fixture tests, converted from build_test_structures.py
(Phase 4.5) into tmp_path-based pytest fixtures.

Each case encodes a finding already confirmed by hand (see development.md):
  - Case 1: package-style local imports resolve correctly; sys.path.append-style
    dynamic imports do not (a known, deliberately deferred limitation).
  - Case 2: local-module resolution is scoped to the --batch root's immediate
    contents, not recursive -- an asymmetry with notebook discovery.
  - Case 3: --output-dir avoids collisions between same-stem notebooks in
    sibling directories.
  - Case 4: --output-dir mirrors the notebook itself but does not copy
    non-notebook sibling assets (e.g. data files) -- a deliberate scope
    boundary, not a bug.
"""

import json
import sys
from pathlib import Path

import pytest

import notebook_env as ne


KERNEL_META = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}


def make_notebook(path: Path, code_lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "outputs": [],
                "source": code_lines,
            }
        ],
        "metadata": KERNEL_META,
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")


def make_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.fixture
def minimal_environment(monkeypatch):
    """No packages pre-installed; isolates local-module resolution from
    unrelated missing/matched-package noise, and prevents any real
    subprocess calls (pip list / pip freeze) during generation."""
    monkeypatch.setattr(ne, "get_installed_environment", lambda: ({}, []))
    monkeypatch.setattr(ne, "resolve_opencv_variant", lambda submodules=None: "opencv-python")
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **k: pytest.fail("Unexpected real subprocess.run call during generation"),
    )


# ---------------------------------------------------------------------------
# Case builders
# ---------------------------------------------------------------------------

def build_case1(tmp_path: Path) -> Path:
    """Subdirectory helper: package-style vs sys.path.append-style import."""
    project = tmp_path / "project"
    make_file(project / "code" / "helper.py", "def do_thing():\n    return 42\n")
    make_notebook(
        project / "train_package_style.ipynb",
        ["from code import helper\n", "helper.do_thing()\n"],
    )
    make_notebook(
        project / "train_syspath_style.ipynb",
        ["import sys\n", "sys.path.append('code')\n", "import helper\n", "helper.do_thing()\n"],
    )
    return project


def build_case2(tmp_path: Path) -> Path:
    """Root-level helper: repo/common.py + repo/notebooks/run.ipynb.
    Returns the case root (parent of repo/); caller picks --batch target."""
    case_root = tmp_path / "case2_root_helper"
    make_file(case_root / "repo" / "common.py", "def shared():\n    return 1\n")
    make_notebook(
        case_root / "repo" / "notebooks" / "run.ipynb",
        ["import common\n", "common.shared()\n"],
    )
    return case_root


def build_case3(tmp_path: Path) -> Path:
    """Duplicate stems across sibling directories."""
    root = tmp_path / "case3_duplicate_stems"
    make_notebook(root / "dir_a" / "pipeline.ipynb", ["import pandas as pd\n", "# dir_a variant\n"])
    make_notebook(root / "dir_b" / "pipeline.ipynb", ["import numpy as np\n", "# dir_b variant\n"])
    return root


def build_case4(tmp_path: Path) -> Path:
    """Relative asset dependency: notebook reads a sibling data file."""
    root = tmp_path / "case4_relative_assets"
    make_file(root / "analysis" / "data" / "data.csv", "a,b\n1,2\n")
    make_notebook(
        root / "analysis" / "run.ipynb",
        ["import pandas as pd\n", "df = pd.read_csv('data/data.csv')\n"],
    )
    return root


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

def assert_case1(out: str, **_) -> None:
    missing = _missing_block(out)
    assert "train_package_style.ipynb" not in missing, (
        "package-style 'from code import helper' should resolve as local, "
        "so its notebook should not appear anywhere in the missing-packages report"
    )
    # sys.path.append-style import is a confirmed, deliberately-deferred gap:
    # only the directory name ('code') is registered as local, not 'helper' itself.
    assert "helper" in missing and "train_syspath_style.ipynb" in missing, (
        "expected 'helper' flagged missing, attributed to the sys.path.append notebook"
    )


def assert_case2_narrow_batch(out: str, **_) -> None:
    assert "common" in _missing_block(out), (
        "pointing --batch at the parent of repo/ should NOT resolve 'common' "
        "(local-module resolution is not recursive)"
    )


def assert_case2_repo_batch(out: str, **_) -> None:
    assert "common" not in _missing_block(out), (
        "pointing --batch directly at repo/ should resolve 'common' as local"
    )


def assert_case3(out: str, output_dir: Path, **_) -> None:
    out_a = output_dir / "dir_a" / "pipeline.ipynb"
    out_b = output_dir / "dir_b" / "pipeline.ipynb"
    assert out_a.exists() and out_b.exists(), "both same-stem notebooks should be mirrored without collision"
    text_a = out_a.read_text(encoding="utf-8")
    text_b = out_b.read_text(encoding="utf-8")
    assert "dir_a variant" in text_a and "dir_b variant" in text_b, "outputs should retain their distinct source content"


def assert_case4(out: str, output_dir: Path, **_) -> None:
    mirrored_nb = output_dir / "analysis" / "run.ipynb"
    mirrored_csv = output_dir / "analysis" / "data" / "data.csv"
    assert mirrored_nb.exists(), "the notebook itself should be mirrored to --output-dir"
    assert not mirrored_csv.exists(), (
        "non-notebook sibling assets are a deliberate scope boundary and should NOT be copied; "
        "if this now passes, either a real feature was added (update this test) or something regressed"
    )


def _missing_block(out: str) -> str:
    """Isolate the 'Packages missing from current environment' section so
    matches don't accidentally hit an unrelated part of the report."""
    marker = "Packages missing from current environment"
    idx = out.find(marker)
    return out[idx:] if idx != -1 else ""


# ---------------------------------------------------------------------------
# Scenarios: (build_fn, cli_args_fn, assert_fn)
# cli_args_fn(built_path, output_dir) -> list[str] of argv after the script name
# ---------------------------------------------------------------------------

SCENARIOS = {
    "case1_subdir_helper": (
        build_case1,
        lambda built, out_dir: ["--batch", str(built), "--analyze"],
        assert_case1,
    ),
    "case2_narrow_batch": (
        build_case2,
        lambda built, out_dir: ["--batch", str(built), "--analyze"],
        assert_case2_narrow_batch,
    ),
    "case2_repo_batch": (
        build_case2,
        lambda built, out_dir: ["--batch", str(built / "repo"), "--analyze"],
        assert_case2_repo_batch,
    ),
    "case3_duplicate_stems": (
        build_case3,
        lambda built, out_dir: ["--batch", str(built), "--output-dir", str(out_dir)],
        assert_case3,
    ),
    "case4_relative_assets": (
        build_case4,
        lambda built, out_dir: ["--batch", str(built), "--output-dir", str(out_dir)],
        assert_case4,
    ),
}


@pytest.mark.parametrize("scenario_id", list(SCENARIOS.keys()))
def test_structural_case(scenario_id, tmp_path, minimal_environment, monkeypatch, capsys):
    build_fn, cli_args_fn, assert_fn = SCENARIOS[scenario_id]

    built = build_fn(tmp_path)
    output_dir = tmp_path / "output"

    argv = ["notebook_env.py"] + cli_args_fn(built, output_dir)
    monkeypatch.setattr(sys, "argv", argv)

    # Batch mode's exit behavior on success isn't what this test is about --
    # tolerate either a clean return or sys.exit(0), fail only on a real error exit.
    try:
        ne.main()
    except SystemExit as exc:
        assert exc.code in (0, None), f"unexpected non-zero exit: {exc.code}"

    out = capsys.readouterr().out

    assert_fn(out, output_dir=output_dir)
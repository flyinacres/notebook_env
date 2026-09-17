#!/usr/bin/env python3
"""Phase 5f: Hardware Mocking Test.

Generates a mock PyTorch backend package dynamically, evaluates notebook_env against
a test notebook, and verifies proper hardware detection in generated Markdown metadata.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

from e2e_harness import (
    FIXTURES_DIR,
    fail_test,
    get_cell_source,
    load_notebook,
    run_notebook_env,
    temp_directory,
    temp_notebook,
)

MOCK_TORCH_CODE = """
import os
__version__ = "2.3.0+mock"
class _CUDA:
    @staticmethod
    def is_available(): return os.environ.get("MOCK_CUDA_AVAILABLE", "0") == "1"
    @staticmethod
    def get_device_name(device=None): return "NVIDIA A100-SXM4-40GB (Mock)"
class _MPS:
    @staticmethod
    def is_available(): return os.environ.get("MOCK_MPS_AVAILABLE", "0") == "1"
    @staticmethod
    def is_built(): return os.environ.get("MOCK_MPS_AVAILABLE", "0") == "1"
class _Backends:
    mps = _MPS()
cuda = _CUDA()
backends = _Backends()
"""


def main() -> None:
    mode = os.environ.get("TEST_HW_MODE", "none").lower()
    fixture_path = FIXTURES_DIR / "temp_hw_fixture.ipynb"
    mock_base = FIXTURES_DIR / "mock_pkgs"
    mock_pkg_dir = mock_base / "torch"

    print(f"1. Testing hardware mode: {mode.upper()}...")

    with temp_directory(mock_base):
        mock_pkg_dir.mkdir(parents=True, exist_ok=True)
        (mock_pkg_dir / "__init__.py").write_text(MOCK_TORCH_CODE, encoding="utf-8")

        with temp_notebook(fixture_path, ["import torch"]):
            env_override = {"PYTHONPATH": f"{mock_base}:{os.environ.get('PYTHONPATH', '')}"}
            result = run_notebook_env(str(fixture_path), "--in-place", env=env_override)

            if not result.ok:
                fail_test(
                    "Execute notebook_env for Mock Hardware",
                    f"Process exited with non-zero status {result.returncode}",
                    stdout=result.stdout,
                    stderr=result.stderr,
                    details={"mode": mode},
                )

            nb_data = load_notebook(fixture_path)
            md_cell = get_cell_source(nb_data, 0).lower()

            if mode == "cuda":
                if "a100" not in md_cell and "cuda" not in md_cell:
                    fail_test(
                        "Verify CUDA Markdown Metadata",
                        "CUDA hardware indicators not found in generated setup cell.",
                        details={"captured_markdown": md_cell},
                    )
                print("   PASS: CUDA hardware correctly detected and documented.")

            elif mode == "mps":
                if "mps" not in md_cell and "apple" not in md_cell and "metal" not in md_cell:
                    fail_test(
                        "Verify Apple Silicon MPS Metadata",
                        "MPS hardware indicators not found in generated setup cell.",
                        details={"captured_markdown": md_cell},
                    )
                print("   PASS: Apple Silicon (MPS) hardware correctly detected and documented.")


if __name__ == "__main__":
    main()
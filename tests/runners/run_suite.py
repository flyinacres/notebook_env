#!/usr/bin/env python3
"""E2E Test Suite Runner for notebook_env.

Executes Dockerized end-to-end integration tests across target runtime tiers
(python3.11, kaggle, colab, local_pkg) and common tier-independent tests.

Examples:
    python tests/runners/run_suite.py --tier common
    python tests/runners/run_suite.py --tier python3.11
    python tests/runners/run_suite.py --all
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Sequence

# Repository root (two levels up from tests/runners/)
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRATCH_DIR = REPO_ROOT / ".test_artifacts"
ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


@dataclass(frozen=True)
class PositiveFixture:
    path: str
    verify_patterns: tuple[str, ...] = ()


@dataclass(frozen=True)
class NegativeFixture:
    path: str
    expected_ename: str
    expected_evalue_substring: str


@dataclass(frozen=True)
class TierConfig:
    image: str
    positive_fixtures: tuple[PositiveFixture, ...] = ()
    negative_fixtures: tuple[NegativeFixture, ...] = ()


TIER_CONFIGS: dict[str, TierConfig] = {
    "python3.11": TierConfig(
        image="python:3.11-slim",
        positive_fixtures=(
            PositiveFixture(
                path="tests/fixtures/e2e/test_partial_install_recovery.ipynb",
                verify_patterns=(
                    "Partial install succeeded for valid packages: humanize==4.16.0, tabulate==0.9.0",
                    "tabulate==0.0.0.nonexistent failed to install",
                ),
            ),
            PositiveFixture(
                path="tests/fixtures/e2e/test_numpy_old_pin_preserves_api.ipynb",
                verify_patterns=(
                    "numpy==1.23.5: numpy.bool alias still works as expected",
                ),
            ),
        ),
        negative_fixtures=(
            NegativeFixture(
                path="tests/fixtures/e2e/test_e2e_failed_repin_surfaces_downstream.ipynb",
                expected_ename="AssertionError",
                expected_evalue_substring="failed silently",
            ),
        ),
    ),
    "kaggle": TierConfig(
        image="gcr.io/kaggle-images/python:latest",
        positive_fixtures=(
            PositiveFixture(path="tests/fixtures/e2e/test_pip_satisfied.ipynb"),
            PositiveFixture(path="tests/fixtures/e2e/pinned_install.ipynb"),
            PositiveFixture(path="tests/fixtures/e2e/platform_pseudo_module.ipynb"),
        ),
    ),
    "colab": TierConfig(
        image="us-docker.pkg.dev/colab-images/public/cpu-runtime:latest",
        positive_fixtures=(
            PositiveFixture(path="tests/fixtures/e2e/test_pip_satisfied.ipynb"),
            PositiveFixture(path="tests/fixtures/e2e/pinned_install.ipynb"),
            PositiveFixture(path="tests/fixtures/e2e/platform_pseudo_module.ipynb"),
        ),
    ),
    "local_pkg": TierConfig(
        image="python:3.11-slim",
        positive_fixtures=(
            PositiveFixture(
                path="tests/fixtures/e2e/test_local_pkg_timeout.ipynb",
                verify_patterns=("Installation timed out after 2s.",),
            ),
            PositiveFixture(
                path="tests/fixtures/e2e/test_local_pkg_build_failure.ipynb",
                verify_patterns=(
                    "RuntimeError: deliberate build failure for local_test_pkg",
                ),
            ),
        ),
    ),
}


def build_docker_cmd(tier: str, notebook_path: str, merged_path: str, output_path: str) -> str:
    """Build the internal bash shell commands for each tier environment."""
    if tier == "python3.11":
        return (
            "pip install --no-cache-dir ipykernel nbconvert==7.17.1 humanize==4.16.0 "
            "tabulate==0.9.0 numpy==1.23.5 packaging resolvelib && "
            "python -m ipykernel install --user --name python3 && "
            f'python notebook_env.py "{notebook_path}" --output && '
            f'jupyter nbconvert --to notebook --execute "{merged_path}" --output "{output_path}" '
            "--ExecutePreprocessor.timeout=300 --ExecutePreprocessor.kernel_name=python3"
        )
    if tier == "kaggle":
        return (
            "pip install --quiet packaging resolvelib && "
            "python3 -m venv --system-site-packages --without-pip --clear /tmp/run_env && "
            f'/tmp/run_env/bin/python notebook_env.py "{notebook_path}" --output && '
            f'/tmp/run_env/bin/python -m jupyter nbconvert --to notebook --execute "{merged_path}" '
            f'--output "{output_path}" --ExecutePreprocessor.timeout=300'
        )
    if tier == "colab":
        return (
            "pip install --quiet packaging resolvelib && "
            f'python3 notebook_env.py "{notebook_path}" --output && '
            f'jupyter nbconvert --to notebook --execute "{merged_path}" --output "{output_path}" '
            "--ExecutePreprocessor.timeout=300 --ExecutePreprocessor.kernel_name=python3"
        )
    if tier == "local_pkg":
        return (
            "pip install --no-cache-dir ipykernel nbconvert==7.17.1 packaging resolvelib -q && "
            "python -m ipykernel install --user --name python3 && "
            "PIP_NO_INDEX=1 PIP_FIND_LINKS=/workspace/tests/fixtures/local_test_pkg/bootstrap "
            "pip install --no-cache-dir setuptools wheel && "
            f'SEED_VER=$(grep -oE "local_test_pkg==[0-9.]+" "{notebook_path}" | head -1 | cut -d= -f3) && '
            "PIP_NO_INDEX=1 PIP_FIND_LINKS=/workspace/tests/fixtures/local_test_pkg/seed_dist "
            'pip install --no-cache-dir local_test_pkg=="$SEED_VER" && '
            f'python notebook_env.py "{notebook_path}" --output --timeout 2 && '
            "pip uninstall -y local_test_pkg && "
            "PIP_NO_INDEX=1 PIP_FIND_LINKS=/workspace/tests/fixtures/local_test_pkg/dist "
            "PIP_NO_BUILD_ISOLATION=1 PIP_NO_CACHE_DIR=1 "
            f'jupyter nbconvert --to notebook --execute "{merged_path}" --output "{output_path}" '
            "--ExecutePreprocessor.timeout=300 --ExecutePreprocessor.kernel_name=python3"
        )
    raise ValueError(f"No Docker command mapping configured for tier: {tier}")


def run_docker(
    image: str,
    bash_script: str,
    env: dict[str, str] | None = None,
) -> tuple[int, str]:
    """Execute a bash command inside a container, streaming output and capturing text."""
    cmd = [
        "docker",
        "run",
        "--rm",
        "--pull",
        "missing",
        "-v",
        f"{REPO_ROOT}:/workspace",
        "-w",
        "/workspace",
        "-e",
        "PYTHONPATH=/workspace",
        "-e",
        "PYTHONUNBUFFERED=1",
        "-e",
        "PYDEVD_DISABLE_FILE_VALIDATION=1",
        "-e",
        "PIP_ROOT_USER_ACTION=ignore",
    ]

    if env:
        for k, v in env.items():
            cmd.extend(["-e", f"{k}={v}"])

    cmd.extend([
        "--entrypoint",
        "/bin/bash",
        image,
        "-c",
        f"{bash_script} 2>&1",
    ])

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    lines: list[str] = []
    if process.stdout:
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lines.append(line)

    process.wait()
    return process.returncode, "".join(lines)


def cleanup_artifacts(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            try:
                path.unlink()
            except OSError as err:
                print(f"Warning: Failed to remove artifact {path}: {err}", file=sys.stderr)


def verify_positive_notebook(notebook_path: Path, patterns: Sequence[str]) -> None:
    if not notebook_path.exists():
        raise RuntimeError(f"Expected output notebook was not generated: {notebook_path}")

    with notebook_path.open("r", encoding="utf-8") as f:
        nb_data = json.load(f)

    collected_text: list[str] = []
    for cell_idx, cell in enumerate(nb_data.get("cells", [])):
        for output in cell.get("outputs", []):
            output_type = output.get("output_type")
            if output_type == "error":
                ename = output.get("ename", "Error")
                evalue = output.get("evalue", "")
                raise AssertionError(
                    f"Unexpected cell execution error in {notebook_path.name} (cell {cell_idx}): {ename}: {evalue}"
                )
            if output_type == "stream":
                text = output.get("text", "")
                collected_text.append("".join(text) if isinstance(text, list) else text)
            elif output_type in {"execute_result", "display_data"}:
                data_text = output.get("data", {}).get("text/plain", "")
                collected_text.append("".join(data_text) if isinstance(data_text, list) else data_text)

    if not patterns:
        return

    all_output = strip_ansi("\n".join(collected_text))
    for pattern in patterns:
        if pattern not in all_output:
            raise AssertionError(
                f"Verification failed for {notebook_path.name}.\n"
                f"Missing expected pattern:\n  '{pattern}'\n"
                f"Searched outputs:\n{all_output}"
            )

def verify_negative_notebook(
    notebook_path: Path,
    expected_ename: str,
    expected_evalue_substring: str,
) -> None:
    if not notebook_path.exists():
        raise RuntimeError(f"Expected output notebook was not generated: {notebook_path}")

    with notebook_path.open("r", encoding="utf-8") as f:
        nb_data = json.load(f)

    error_outputs: list[dict] = []
    for cell_idx, cell in enumerate(nb_data.get("cells", [])):
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                error_outputs.append({"cell": cell_idx, "output": output})

    if len(error_outputs) != 1:
        raise AssertionError(
            f"Expected exactly one cell error in {notebook_path.name}, but found {len(error_outputs)}."
        )

    err = error_outputs[0]["output"]
    ename = err.get("ename", "")
    evalue = err.get("evalue", "")

    if ename != expected_ename:
        raise AssertionError(
            f"Cell error ename mismatch: expected '{expected_ename}', found '{ename}'"
        )
    if expected_evalue_substring not in evalue:
        raise AssertionError(
            f"Cell error evalue mismatch: expected substring '{expected_evalue_substring}', found '{evalue}'"
        )


def run_common_tests() -> None:
    print("=" * 60)
    print(" Running tier-independent tests")
    print("=" * 60)

    print("\033[96mRunning Check-Drift E2E Test (generate + check-drift subprocesses)...\033[0m")
    exit_code, _ = run_docker(
        "python:3.11-slim",
        "pip install --no-cache-dir packaging resolvelib -q && python tests/runners/test_check_drift.py",
    )
    if exit_code != 0:
        raise RuntimeError("Check-drift e2e test failed.")
    print("\033[92mPASS: Check-drift e2e test\n\033[0m")

    print("\033[96mRunning Phase 5g: Live-Kernel Stale Module Test...\033[0m")
    exit_code, _ = run_docker(
        "python:3.11-slim",
        "pip install --no-cache-dir packaging resolvelib jupyter_client ipykernel numpy==1.26.4 -q && "
        "python -m ipykernel install --user --name python3 && "
        "python tests/runners/test_live_kernel_stale_repin.py",
    )
    if exit_code != 0:
        raise RuntimeError("Live-kernel stale module test failed.")
    print("\033[92mPASS: Live-kernel stale module test\n\033[0m")

    print("\033[96mRunning Phase 5g: Live-Kernel Phase 0 Regressions...\033[0m")
    exit_code, _ = run_docker(
        "python:3.11-slim",
        "pip install --no-cache-dir packaging resolvelib jupyter_client ipykernel packaging resolvelib -q && "
        "python -m ipykernel install --user --name python3 && "
        "python tests/runners/test_live_kernel_phase0_regressions.py",
    )
    if exit_code != 0:
        raise RuntimeError("Live-kernel Phase 0 regressions test failed.")
    print("\033[92mPASS: Live-kernel Phase 0 regressions test\n\033[0m")

    print("\033[96mRunning Phase 5f: Hardware Mocking (CUDA)...\033[0m")
    exit_code, _ = run_docker(
        "python:3.11-slim",
        "pip install --no-cache-dir packaging resolvelib -q && python tests/runners/test_hardware_mock.py",
        env={
            "PYTHONPATH": "/workspace/tests/fixtures/mock_pkgs:/workspace",
            "TEST_HW_MODE": "cuda",
            "MOCK_CUDA_AVAILABLE": "1",
            "MOCK_MPS_AVAILABLE": "0",
        },
    )
    if exit_code != 0:
        raise RuntimeError("Hardware mocking test (CUDA) failed.")

    print("\033[96mRunning Phase 5f: Hardware Mocking (MPS)...\033[0m")
    exit_code, _ = run_docker(
        "python:3.11-slim",
        "pip install --no-cache-dir packaging resolvelib -q && python tests/runners/test_hardware_mock.py",
        env={
            "PYTHONPATH": "/workspace/tests/fixtures/mock_pkgs:/workspace",
            "TEST_HW_MODE": "mps",
            "MOCK_CUDA_AVAILABLE": "0",
            "MOCK_MPS_AVAILABLE": "1",
        },
    )
    if exit_code != 0:
        raise RuntimeError("Hardware mocking test (MPS) failed.")
    print("\033[92mPASS: Hardware mocking tests\n\033[0m")

    print("\033[96mRunning Phase 5k: Local Package Pin-and-Verify (1.0.0 -> 2.0.0)...\033[0m")
    repin_nb = Path("tests/fixtures/e2e/test_local_pkg_pin_and_verify.ipynb")
    repin_merged = repin_nb.with_name(repin_nb.stem + "_merged.ipynb")
    out_v1 = SCRATCH_DIR / "out_v1.ipynb"
    out_v2 = SCRATCH_DIR / "out_v2.ipynb"
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

    repin_cmd = (
        "pip install --no-cache-dir ipykernel nbconvert==7.17.1 packaging resolvelib -q && "
        "python -m ipykernel install --user --name python3 && "
        "PIP_NO_INDEX=1 PIP_FIND_LINKS=/workspace/tests/fixtures/local_test_pkg/dist "
        "pip install --no-cache-dir local_test_pkg==1.0.0 && "
        f"PIP_NO_INDEX=1 PIP_FIND_LINKS=/workspace/tests/fixtures/local_test_pkg/dist "
        f'python notebook_env.py "{repin_nb.as_posix()}" --output && '
        "pip uninstall -y local_test_pkg && "
        "PIP_NO_INDEX=1 PIP_FIND_LINKS=/workspace/tests/fixtures/local_test_pkg/dist "
        f'jupyter nbconvert --to notebook --execute "{repin_merged.as_posix()}" '
        f'--output "/workspace/{out_v1.relative_to(REPO_ROOT).as_posix()}" '
        "--ExecutePreprocessor.timeout=300 --ExecutePreprocessor.kernel_name=python3 && "
        "PIP_NO_INDEX=1 PIP_FIND_LINKS=/workspace/tests/fixtures/local_test_pkg/dist "
        "pip install --no-cache-dir local_test_pkg==2.0.0 && "
        f"PIP_NO_INDEX=1 PIP_FIND_LINKS=/workspace/tests/fixtures/local_test_pkg/dist "
        f'python notebook_env.py "{repin_nb.as_posix()}" --output && '
        "pip uninstall -y local_test_pkg && "
        "PIP_NO_INDEX=1 PIP_FIND_LINKS=/workspace/tests/fixtures/local_test_pkg/dist "
        f'jupyter nbconvert --to notebook --execute "{repin_merged.as_posix()}" '
        f'--output "/workspace/{out_v2.relative_to(REPO_ROOT).as_posix()}" '
        "--ExecutePreprocessor.timeout=300 --ExecutePreprocessor.kernel_name=python3"
    )

    try:
        exit_code, _ = run_docker("python:3.11-slim", repin_cmd)
        if exit_code != 0:
            raise RuntimeError("Local package pin-and-verify run failed.")

        verify_positive_notebook(out_v1, ("Pinned install verification passed: local_test_pkg 1.0.0 active",))
        verify_positive_notebook(out_v2, ("Pinned install verification passed: local_test_pkg 2.0.0 active",))
    finally:
        cleanup_artifacts(REPO_ROOT / repin_merged, out_v1, out_v2)

    print("\033[92mPASS: Local package pin-and-verify test\n\033[0m")


def run_tier_tests(tier: str) -> None:
    config = TIER_CONFIGS[tier]
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f" Running E2E Suite: {tier.upper()} TIER ({config.image})")
    print("=" * 60)

    for fixture in config.positive_fixtures:
        nb_path = Path(fixture.path)
        merged_path = nb_path.with_name(nb_path.stem + "_merged.ipynb")
        output_name = f"out_{tier}_{nb_path.stem}.ipynb"
        output_host_path = SCRATCH_DIR / output_name
        output_container_path = f"/workspace/{output_host_path.relative_to(REPO_ROOT).as_posix()}"

        print(f"\n=== [{tier.upper()}] Processing (Expect PASS): {nb_path.as_posix()} ===")
        cmd = build_docker_cmd(tier, nb_path.as_posix(), merged_path.as_posix(), output_container_path)

        try:
            exit_code, _ = run_docker(config.image, cmd)
            if exit_code != 0:
                raise RuntimeError(f"Tier execution failed on positive test {nb_path.as_posix()} (exit code {exit_code})")
            verify_positive_notebook(output_host_path, fixture.verify_patterns)
        finally:
            cleanup_artifacts(REPO_ROOT / merged_path, output_host_path)

        print(f">>> PASS: {nb_path.as_posix()}")

    for fixture in config.negative_fixtures:
        nb_path = Path(fixture.path)
        merged_path = nb_path.with_name(nb_path.stem + "_merged.ipynb")
        output_name = f"out_{tier}_{nb_path.stem}.ipynb"
        output_host_path = SCRATCH_DIR / output_name
        output_container_path = f"/workspace/{output_host_path.relative_to(REPO_ROOT).as_posix()}"

        print(f"\n=== [{tier.upper()}] Processing (Expect FAIL): {nb_path.as_posix()} ===")
        base_cmd = build_docker_cmd(tier, nb_path.as_posix(), merged_path.as_posix(), output_container_path)
        cmd = f"{base_cmd} --allow-errors"

        try:
            exit_code, _ = run_docker(config.image, cmd)
            if exit_code != 0:
                raise RuntimeError(
                    f"Tier container crashed unexpectedly during negative test execution for {nb_path.as_posix()} (exit code {exit_code})"
                )
            verify_negative_notebook(
                output_host_path,
                expected_ename=fixture.expected_ename,
                expected_evalue_substring=fixture.expected_evalue_substring,
            )
        finally:
            cleanup_artifacts(REPO_ROOT / merged_path, output_host_path)

        print(f">>> PASS (Structural failure verified): {nb_path.as_posix()}")

    print("\n" + "=" * 60)
    print(f" {tier.upper()} TIER: ALL TESTS PASSED")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run E2E Dockerized integration suites across target environments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--tier",
        choices=["common", *TIER_CONFIGS.keys()],
        help="Execute a specific tier suite ('common' for live-kernel/hardware mock tests).",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Run common tests followed by all tier configurations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.all:
            run_common_tests()
            for tier in TIER_CONFIGS:
                run_tier_tests(tier)
        elif args.tier == "common":
            run_common_tests()
        else:
            run_tier_tests(args.tier)

        print("\n" + "=" * 60)
        print(" ALL REQUESTED TESTS PASSED")
        print("=" * 60)
    except Exception as err:
        print(f"\n\033[91mFAILED: {err}\033[0m", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
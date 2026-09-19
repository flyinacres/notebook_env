#!/usr/bin/env python3
"""Shared test harness and utilities for notebook_env end-to-end runners.

Provides standardized subprocess wrappers, interactive Jupyter kernel management,
safe fixture lifecycle contexts, and formatted diagnostic failure output.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
from typing import Any, Generator, Sequence

# Canonical project paths
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = WORKSPACE_ROOT / "tests" / "fixtures"


@dataclass(frozen=True)
class SubprocessResult:
    """Captured result of an executed CLI or tool command."""
    command: list[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


@dataclass(frozen=True)
class KernelExecutionResult:
    """Captured result from an interactive Jupyter kernel evaluation."""
    stdout: str
    errors: list[str]

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


def fail_test(
    step_name: str,
    reason: str,
    *,
    stdout: str | None = None,
    stderr: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    """Prints a structured, high-visibility diagnostic failure block and exits 1."""
    border = "=" * 70
    subborder = "-" * 70
    sys.stderr.write(f"\n{border}\n")
    sys.stderr.write(f"TEST STEP FAILURE: {step_name}\n")
    sys.stderr.write(f"REASON: {reason}\n")
    sys.stderr.write(f"{subborder}\n")

    if details:
        sys.stderr.write("METADATA / VARIABLES:\n")
        for key, val in details.items():
            sys.stderr.write(f"  {key}: {val}\n")
        sys.stderr.write(f"{subborder}\n")

    if stdout and stdout.strip():
        sys.stderr.write("CAPTURED STDOUT:\n")
        sys.stderr.write(stdout.rstrip() + "\n")
        sys.stderr.write(f"{subborder}\n")

    if stderr and stderr.strip():
        sys.stderr.write("CAPTURED STDERR:\n")
        sys.stderr.write(stderr.rstrip() + "\n")
        sys.stderr.write(f"{subborder}\n")

    sys.stderr.write(f"{border}\n\n")
    sys.stderr.flush()
    sys.exit(1)


def run_cli_command(
    args: Sequence[str],
    *,
    env: dict[str, str] | None = None,
    cwd: Path | str | None = None,
) -> SubprocessResult:
    """Runs a command via subprocess and captures stdout and stderr cleanly."""
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    proc = subprocess.run(
        list(args),
        cwd=str(cwd) if cwd else str(WORKSPACE_ROOT),
        env=merged_env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return SubprocessResult(
        command=list(args),
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def run_notebook_env(
    *tool_args: str,
    env: dict[str, str] | None = None,
) -> SubprocessResult:
    """Invokes notebook_env.py using the current Python interpreter."""
    cmd = [sys.executable, "notebook_env.py", *tool_args]
    return run_cli_command(cmd, env=env)


@contextmanager
def temp_notebook(
    path: Path | str,
    code_cells: Sequence[str | list[str]],
    *,
    metadata: dict[str, Any] | None = None,
) -> Generator[Path, None, None]:
    """Generates a valid nbformat 4.5 notebook and guarantees its deletion on exit."""
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    formatted_cells = []
    for cell in code_cells:
        lines = cell if isinstance(cell, list) else [cell]
        formatted_cells.append({
            "cell_type": "code",
            "source": lines,
            "metadata": {},
            "outputs": [],
            "execution_count": None,
        })

    nb_data = {
        "cells": formatted_cells,
        "metadata": metadata or {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    with target_path.open("w", encoding="utf-8") as f:
        json.dump(nb_data, f, indent=2)

    try:
        yield target_path
    finally:
        if target_path.exists():
            target_path.unlink()


@contextmanager
def temp_directory(path: Path | str) -> Generator[Path, None, None]:
    """Ensures a directory tree is created and completely removed on exit."""
    target_path = Path(path)
    target_path.mkdir(parents=True, exist_ok=True)
    try:
        yield target_path
    finally:
        if target_path.exists():
            shutil.rmtree(target_path, ignore_errors=True)


def load_notebook(path: Path | str) -> dict[str, Any]:
    """Safely loads and parses a notebook JSON file with clear diagnostics."""
    target_path = Path(path)
    if not target_path.exists():
        fail_test(
            "Load Notebook",
            f"File does not exist: {target_path}",
            details={"path": str(target_path)},
        )

    try:
        with target_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        fail_test(
            "Load Notebook",
            f"Failed to parse JSON notebook structure: {exc}",
            details={"path": str(target_path)},
        )


def get_cell(notebook_data: dict[str, Any], index: int) -> dict[str, Any]:
    """Extracts a cell by index with boundary validation."""
    cells = notebook_data.get("cells", [])
    if index < 0 or index >= len(cells):
        fail_test(
            "Extract Cell",
            f"Cell index {index} out of range (notebook contains {len(cells)} cells).",
            details={"requested_index": index, "total_cells": len(cells)},
        )
    return cells[index]


def get_cell_source(notebook_data: dict[str, Any], index: int) -> str:
    """Extracts and concatenates the source lines of a specific cell."""
    cell = get_cell(notebook_data, index)
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


class InteractiveKernel:
    """Wrapper around jupyter_client providing clean execution and output capture."""

    def __init__(self, kernel_name: str = "python3") -> None:
        self.kernel_name = kernel_name
        self._km: Any = None
        self._client: Any = None
        self._devnull: Any = None

    def start(self, ready_timeout: int = 10, quiet: bool = False) -> None:
        """quiet=True discards the kernel process's own stderr (for example ipykernel's
        "running over TCP without encryption" warning). Errors raised by executed code
        still arrive through execute()."""
        import jupyter_client
        self._km = jupyter_client.KernelManager(kernel_name=self.kernel_name)
        if quiet:
            self._devnull = open(os.devnull, "w")
            self._km.start_kernel(stderr=self._devnull)
        else:
            self._km.start_kernel()
        self._client = self._km.client()
        self._client.start_channels()
        self._client.wait_for_ready(timeout=ready_timeout)

    def shutdown(self) -> None:
        if self._client:
            self._client.stop_channels()
            self._client = None
        if self._km:
            self._km.shutdown_kernel()
            self._km = None
        if self._devnull:
            self._devnull.close()
            self._devnull = None

    def execute(self, code: str, timeout: int = 30) -> KernelExecutionResult:
        """Executes code in the active kernel and captures stdout and errors."""
        if not self._client:
            raise RuntimeError("Kernel is not active. Use within 'interactive_kernel()' context.")

        msg_id = self._client.execute(code)
        outputs: list[str] = []
        errors: list[str] = []

        while True:
            try:
                msg = self._client.get_iopub_msg(timeout=timeout)
                if msg.get("parent_header", {}).get("msg_id") != msg_id:
                    continue

                msg_type = msg.get("msg_type")
                content = msg.get("content", {})

                if msg_type == "stream":
                    outputs.append(content.get("text", ""))
                elif msg_type in ("execute_result", "display_data"):
                    text_data = content.get("data", {}).get("text/plain", "")
                    if text_data:
                        outputs.append(text_data)
                elif msg_type == "error":
                    ename = content.get("ename", "Error")
                    evalue = content.get("evalue", "")
                    errors.append(f"{ename}: {evalue}")
                elif msg_type == "status" and content.get("execution_state") == "idle":
                    break
            except queue.Empty:
                errors.append(f"TimeoutError: Kernel execution timed out after {timeout} seconds.")
                break

        return KernelExecutionResult(stdout="".join(outputs), errors=errors)


@contextmanager
def interactive_kernel(
    kernel_name: str = "python3",
    ready_timeout: int = 10,
    quiet: bool = False,
) -> Generator[InteractiveKernel, None, None]:
    """Context manager for an interactive kernel session with deterministic shutdown."""
    session = InteractiveKernel(kernel_name=kernel_name)
    session.start(ready_timeout=ready_timeout, quiet=quiet)
    try:
        yield session
    finally:
        session.shutdown()
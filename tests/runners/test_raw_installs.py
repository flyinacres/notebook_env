#!/usr/bin/env python3
"""Raw installs end-to-end: packages that come from somewhere other than PyPI.

Builds a tiny wheel by hand (no build tools, no network), serves it from a local
HTTP server, and drives notebook_env.py plus a real Jupyter kernel through the four
situations that matter. Each one generates a locked notebook, removes the package,
then executes the generated notebook in a fresh kernel:

  1. Explicit path       The notebook itself has `%pip install <wheel path>`. The path is carried
                         verbatim in raw_installs, the package is not pinned as a PyPI package, and
                         Cell 2 reinstalls it in a clean environment.
  2. Inferred URL        The package was installed by hand from a URL and the notebook has no install
                         line. The recorded URL is inferred into raw_installs and Cell 2 reinstalls it.
  3. Unreachable source  The same generated notebook with the server stopped: Cell 2 says so plainly
                         and the failure surfaces downstream instead of being hidden.
  4. Inferred local path Installed by hand from a local path with no install line. Nothing is stored
                         (a path is machine-specific), the path never appears in the output, and Cell 2
                         says it cannot be shared directly.

All pip activity runs with PIP_NO_INDEX=1, so PyPI is never consulted.
"""

from __future__ import annotations

import base64
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import sys
import tempfile
import threading
import zipfile

import platform

from e2e_harness import (
    FIXTURES_DIR,
    WORKSPACE_ROOT,
    fail_test,
    get_cell_source,
    interactive_kernel,
    load_notebook,
    run_cli_command,
    run_notebook_env,
    temp_notebook,
)

sys.path.insert(0, str(WORKSPACE_ROOT))
import notebook_env as ne  # noqa: E402  (needs WORKSPACE_ROOT on sys.path)

DIST_NAME = "rawpkg-probe"
IMPORT_NAME = "rawpkg_probe"
WHEEL_NAME = "rawpkg_probe-1.0.0-py3-none-any.whl"


def verify_code(label: str) -> str:
    return (
        f"import {IMPORT_NAME}\n"
        f"assert {IMPORT_NAME}.VERSION == '1.0.0'\n"
        f"print('RAW-INSTALL-VERIFIED {label}')\n"
    )


def build_wheel(dest_dir: Path) -> Path:
    """A minimal valid wheel, written by hand so the test needs no build tooling."""
    def record_hash(data: bytes) -> str:
        return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()

    dist_info = "rawpkg_probe-1.0.0.dist-info"
    files = {
        f"{IMPORT_NAME}/__init__.py": b'VERSION = "1.0.0"\n',
        f"{dist_info}/METADATA": f"Metadata-Version: 2.1\nName: {DIST_NAME}\nVersion: 1.0.0\n".encode(),
        f"{dist_info}/WHEEL": b"Wheel-Version: 1.0\nGenerator: hand\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    record_lines = [f"{name},{record_hash(data)},{len(data)}" for name, data in files.items()]
    record_lines.append(f"{dist_info}/RECORD,,")
    files[f"{dist_info}/RECORD"] = ("\n".join(record_lines) + "\n").encode()

    dest_dir.mkdir(parents=True, exist_ok=True)
    wheel_path = dest_dir / WHEEL_NAME
    with zipfile.ZipFile(wheel_path, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return wheel_path


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args) -> None:  # keep test output readable
        pass


def start_server(serve_dir: Path) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(serve_dir)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def pip(step: str, *args: str) -> None:
    result = run_cli_command([sys.executable, "-m", "pip", *args])
    if not result.ok:
        fail_test(step, f"pip {' '.join(args)} failed", stdout=result.stdout, stderr=result.stderr)


def remove_package(step: str, verify: bool = False) -> None:
    pip(step, "uninstall", "-y", DIST_NAME)
    if verify:
        shown = run_cli_command([sys.executable, "-m", "pip", "show", DIST_NAME])
        show("pip show", "package not found" if shown.returncode != 0 else "STILL INSTALLED")
        check(step, "package is absent before the notebook runs (a pass cannot come from a leftover install)",
              shown.returncode != 0)


def generate(step: str, code_cells: list[str], name: str):
    """Runs notebook_env.py --output on a temporary notebook. Returns (merged_path, manifest, cell2_source, raw_text)."""
    nb_path = FIXTURES_DIR / f"temp_raw_installs_{name}.ipynb"
    merged_path = nb_path.with_name(nb_path.stem + "_merged.ipynb")
    with temp_notebook(nb_path, code_cells, metadata={"language_info": {"name": "python"}}):
        result = run_notebook_env(str(nb_path), "--output")
        if not result.ok:
            fail_test(step, "notebook_env.py --output failed", stdout=result.stdout, stderr=result.stderr)
        if not merged_path.exists():
            fail_test(step, f"merged notebook was not written: {merged_path}", stdout=result.stdout)
    manifest, error = ne.extract_manifest_from_file(str(merged_path))
    if error:
        fail_test(step, f"could not read the generated manifest: {error}")
    return merged_path, manifest, get_cell_source(load_notebook(merged_path), 1), merged_path.read_text(encoding="utf-8")


def execute(step: str, merged_path: Path) -> tuple[str, list[str], int]:
    """Runs every code cell of a notebook in a fresh kernel. Returns (all stdout, all errors, cells run)."""
    stdout: list[str] = []
    errors: list[str] = []
    cells_run = 0
    with interactive_kernel(ready_timeout=60, quiet=True) as kernel:
        for cell in load_notebook(merged_path)["cells"]:
            if cell.get("cell_type") != "code":
                continue
            source = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
            outcome = kernel.execute(source, timeout=300)
            cells_run += 1
            stdout.append(outcome.stdout)
            errors.extend(outcome.errors)
    return "\n".join(stdout), errors, cells_run


SCRATCH: Path = Path(".")  # set in main(); only used to keep printed paths short
CHECKS_PASSED = 0


RESULTS: list[tuple[str, str]] = []


def line_with(text: str, needle: str) -> str:
    """First line of `text` containing `needle` (evidence for a check)."""
    for line in text.splitlines():
        if needle in line:
            return line.strip()[:200]
    return "(no matching line)"


def tidy(text: object) -> str:
    return str(text).replace(str(SCRATCH), "<scratch>")


def check(step: str, label: str, condition: bool, **details) -> None:
    """Prints a passing check as evidence, or stops the run with the details of a failing one."""
    global CHECKS_PASSED
    if not condition:
        fail_test(step, f"Expected: {label}", details={k: tidy(repr(v)[:1500]) for k, v in details.items()})
    CHECKS_PASSED += 1
    RESULTS.append((step, label))
    print(f"   \u2713 {label}")


def show(heading: str, value: object) -> None:
    print(f"   {heading:<10}: {tidy(value)}")


def show_output(stdout: str, needles: tuple[str, ...], errors: list[str] | None = None) -> None:
    """The lines of kernel output that matter, so a reader can see what actually happened."""
    print("   kernel output (relevant lines):")
    for line in stdout.splitlines():
        if any(n in line for n in needles):
            print(f"     | {tidy(line.strip())[:120]}")
    for err in errors or []:
        print(f"     ! {tidy(err)[:120]}")


def begin(number: int, title: str, setup: str) -> str:
    print(f"\n[{number}/4] {title}")
    print(f"   setup     : {setup}")
    return f"{number}. {title}"


def main() -> None:
    global SCRATCH
    os.environ["PIP_NO_INDEX"] = "1"
    os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"

    SCRATCH = Path(tempfile.mkdtemp(prefix="raw_installs_e2e_"))
    wheel = build_wheel(SCRATCH / "serve")
    server = start_server(SCRATCH / "serve")
    url = f"http://127.0.0.1:{server.server_address[1]}/{wheel.name}"
    merged_files: list[Path] = []
    relevant = ("Installing non-standard sources", "Installing (raw)", "installed successfully",
                "failed to install", "custom-specified source", "contact the notebook", "RAW-INSTALL-VERIFIED")

    print(f"raw_installs end-to-end | Python {platform.python_version()} | PyPI disabled (PIP_NO_INDEX=1)")
    print(f"package   : {DIST_NAME} (hand-built wheel, imports as {IMPORT_NAME})")
    print(f"sources   : local path {tidy(wheel)}")
    print(f"            http URL   {url}")

    try:
        step = begin(1, "Explicit path", "notebook has '%pip install <wheel path>'; package installed by hand first")
        remove_package(step)
        pip(step, "install", str(wheel))
        merged, manifest, cell2, text = generate(step, [f"%pip install {wheel}\n", verify_code("explicit-path")], "explicit")
        merged_files.append(merged)
        show("raw_installs", manifest.raw_installs)
        show("pinned", [d.name for d in manifest.dependencies])
        show("Cell 2 says", line_with(cell2, DIST_NAME))
        check(step, "the notebook's own path is carried verbatim in raw_installs",
              manifest.raw_installs == [str(wheel)], raw_installs=manifest.raw_installs)
        check(step, "the package is not pinned as if it were on PyPI",
              not [d for d in manifest.dependencies if DIST_NAME in d.name], dependencies=manifest.dependencies)
        check(step, "the installed package is not reported as 'not found'",
              "not found via pip-freeze" not in cell2, cell2=cell2[:1500])
        remove_package(step, verify=True)
        stdout, errors, cells = execute(step, merged)
        show("kernel run", f"{cells} code cells executed, {len(errors)} errors")
        show_output(stdout, relevant, errors)
        check(step, "the generated notebook runs cleanly in a fresh kernel", not errors, errors=errors, stdout=stdout)
        check(step, "Cell 2 installs the raw source and says so",
              "Installing non-standard sources" in stdout and f"{wheel} installed successfully" in stdout, stdout=stdout)
        check(step, "the package is usable after Cell 2", "RAW-INSTALL-VERIFIED explicit-path" in stdout, stdout=stdout)

        step = begin(2, "Inferred URL", "installed by hand from a URL; the notebook has no install line")
        remove_package(step)
        pip(step, "install", url)
        merged, manifest, cell2, text = generate(step, [verify_code("inferred-url")], "inferred_url")
        merged_files.append(merged)
        inferred_url_notebook = merged
        show("raw_installs", manifest.raw_installs)
        show("pinned", [d.name for d in manifest.dependencies])
        show("Cell 2 says", line_with(cell2, DIST_NAME))
        check(step, "the recorded source URL is inferred into raw_installs",
              manifest.raw_installs == [url], raw_installs=manifest.raw_installs)
        check(step, "the package is not pinned as if it were on PyPI",
              not [d for d in manifest.dependencies if DIST_NAME in d.name], dependencies=manifest.dependencies)
        check(step, "Cell 2 describes it as installed from a direct URL, not 'not found'",
              "not found via pip-freeze" not in cell2 and "installed from a direct URL" in cell2, cell2=cell2[:1500])
        remove_package(step, verify=True)
        stdout, errors, cells = execute(step, merged)
        show("kernel run", f"{cells} code cells executed, {len(errors)} errors")
        show_output(stdout, relevant, errors)
        check(step, "the generated notebook runs cleanly in a fresh kernel", not errors, errors=errors, stdout=stdout)
        check(step, "Cell 2 installs the inferred URL", f"{url} installed successfully" in stdout, stdout=stdout)
        check(step, "the package is usable after Cell 2", "RAW-INSTALL-VERIFIED inferred-url" in stdout, stdout=stdout)

        step = begin(3, "Unreachable source", "the step 2 notebook again, with the HTTP server stopped")
        server.shutdown()
        server.server_close()
        remove_package(step, verify=True)
        stdout, errors, cells = execute(step, inferred_url_notebook)
        show("kernel run", f"{cells} code cells executed, {len(errors)} errors")
        show_output(stdout, relevant, errors)
        check(step, "Cell 2 reports that the raw install failed", f"{url} failed to install" in stdout, stdout=stdout)
        check(step, "Cell 2 explains what a failed custom source means",
              "custom-specified source" in stdout and "contact the notebook's author" in stdout, stdout=stdout)
        check(step, "the failure surfaces downstream as exactly one ModuleNotFoundError (nothing is hidden)",
              len(errors) == 1 and errors[0].startswith("ModuleNotFoundError"), errors=errors)

        step = begin(4, "Inferred local path", "installed by hand from a local path; the notebook has no install line")
        remove_package(step)
        pip(step, "install", str(wheel))
        merged, manifest, cell2, text = generate(step, [verify_code("local-path")], "local_path")
        merged_files.append(merged)
        show("raw_installs", manifest.raw_installs)
        show("Cell 2 says", line_with(cell2, DIST_NAME))
        show("path search", f"looked for the wheel path and its directory in {len(text)} characters of output")
        check(step, "a machine-specific path is not stored in raw_installs",
              manifest.raw_installs == [], raw_installs=manifest.raw_installs)
        check(step, "the local path appears nowhere in the generated notebook",
              str(wheel) not in text and str(SCRATCH) not in text)
        check(step, "Cell 2 says the package is on a system-dependent path", "system-dependent path" in cell2, cell2=cell2[:1500])
        remove_package(step, verify=True)
        stdout, errors, cells = execute(step, merged)
        show("kernel run", f"{cells} code cells executed, {len(errors)} errors")
        show_output(stdout, relevant, errors)
        check(step, "with nothing to reinstall it, the import fails visibly (exactly one ModuleNotFoundError)",
              len(errors) == 1 and errors[0].startswith("ModuleNotFoundError"), errors=errors, stdout=stdout)

    finally:
        server.shutdown()
        for merged in merged_files:
            if merged.exists():
                merged.unlink()
        run_cli_command([sys.executable, "-m", "pip", "uninstall", "-y", DIST_NAME])

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for scenario in dict.fromkeys(step for step, _ in RESULTS):
        print(f"  {scenario}: {sum(1 for st, _ in RESULTS if st == scenario)} checks passed")
    print(f"\n  Total: {CHECKS_PASSED} checks passed, 0 failed.")

if __name__ == "__main__":
    main()

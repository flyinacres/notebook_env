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


def remove_package(step: str) -> None:
    pip(step, "uninstall", "-y", DIST_NAME)


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


def execute(step: str, merged_path: Path) -> tuple[str, list[str]]:
    """Runs every code cell of a notebook in a fresh kernel. Returns (all stdout, all errors)."""
    stdout: list[str] = []
    errors: list[str] = []
    with interactive_kernel(ready_timeout=60) as kernel:
        for cell in load_notebook(merged_path)["cells"]:
            if cell.get("cell_type") != "code":
                continue
            source = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
            outcome = kernel.execute(source, timeout=300)
            stdout.append(outcome.stdout)
            errors.extend(outcome.errors)
    return "\n".join(stdout), errors


def expect(step: str, condition: bool, reason: str, **details) -> None:
    if not condition:
        fail_test(step, reason, details={k: repr(v)[:1500] for k, v in details.items()})


def main() -> None:
    os.environ["PIP_NO_INDEX"] = "1"
    os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"

    scratch = Path(tempfile.mkdtemp(prefix="raw_installs_e2e_"))
    wheel = build_wheel(scratch / "serve")
    server = start_server(scratch / "serve")
    url = f"http://127.0.0.1:{server.server_address[1]}/{wheel.name}"
    merged_files: list[Path] = []

    try:
        # --- 1. Explicit local path in the notebook ---------------------------------------
        step = "1. Explicit path"
        print(f"{step}: notebook has '%pip install <wheel path>'; generate, remove package, execute...")
        remove_package(step)
        pip(step, "install", str(wheel))
        merged, manifest, cell2, text = generate(step, [f"%pip install {wheel}\n", verify_code("explicit-path")], "explicit")
        merged_files.append(merged)
        expect(step, manifest.raw_installs == [str(wheel)], "the notebook's own path must be carried verbatim in raw_installs",
               raw_installs=manifest.raw_installs)
        expect(step, not [d for d in manifest.dependencies if DIST_NAME in d["name"]],
               "a non-PyPI package must not be pinned as if it were on PyPI", dependencies=manifest.dependencies)
        expect(step, "not found via pip-freeze" not in cell2,
               "an installed package must not be reported as not found", cell2=cell2[:1500])
        remove_package(step)
        stdout, errors = execute(step, merged)
        expect(step, not errors, "the generated notebook raised in a clean environment", errors=errors, stdout=stdout)
        expect(step, "Installing non-standard sources" in stdout and f"{wheel} installed successfully" in stdout,
               "Cell 2 must install the raw source and say so", stdout=stdout)
        expect(step, "RAW-INSTALL-VERIFIED explicit-path" in stdout, "the package was not usable after Cell 2", stdout=stdout)
        print("   PASS")

        # --- 2. Inferred URL: installed by hand, no install line in the notebook ---------------
        step = "2. Inferred URL"
        print(f"{step}: installed by hand from a URL, no install line; generate, remove package, execute...")
        remove_package(step)
        pip(step, "install", url)
        merged, manifest, cell2, text = generate(step, [verify_code("inferred-url")], "inferred_url")
        merged_files.append(merged)
        inferred_url_notebook = merged
        expect(step, manifest.raw_installs == [url], "the recorded source URL must be inferred into raw_installs",
               raw_installs=manifest.raw_installs)
        expect(step, not [d for d in manifest.dependencies if DIST_NAME in d["name"]],
               "a non-PyPI package must not be pinned as if it were on PyPI", dependencies=manifest.dependencies)
        expect(step, "not found via pip-freeze" not in cell2 and "installed from a direct URL" in cell2,
               "Cell 2 must describe the package as installed from a direct URL", cell2=cell2[:1500])
        remove_package(step)
        stdout, errors = execute(step, merged)
        expect(step, not errors, "the generated notebook raised in a clean environment", errors=errors, stdout=stdout)
        expect(step, f"{url} installed successfully" in stdout, "Cell 2 must install the inferred URL", stdout=stdout)
        expect(step, "RAW-INSTALL-VERIFIED inferred-url" in stdout, "the package was not usable after Cell 2", stdout=stdout)
        print("   PASS")

        # --- 3. Unreachable source: same generated notebook, server stopped ---------------------
        step = "3. Unreachable source"
        print(f"{step}: server stopped; execute the notebook from step 2 again...")
        server.shutdown()
        server.server_close()
        remove_package(step)
        stdout, errors = execute(step, inferred_url_notebook)
        expect(step, f"{url} failed to install" in stdout, "Cell 2 must say the raw install failed", stdout=stdout)
        expect(step, "custom-specified source" in stdout and "contact the notebook's author" in stdout,
               "Cell 2 must explain what a failed custom source means", stdout=stdout)
        expect(step, len(errors) == 1 and errors[0].startswith("ModuleNotFoundError"),
               "exactly one downstream ModuleNotFoundError is expected (the failure must not be hidden)", errors=errors)
        print("   PASS")

        # --- 4. Inferred local path: nothing stored, path never leaks ---------------------------
        step = "4. Inferred local path"
        print(f"{step}: installed by hand from a local path, no install line; generate and inspect...")
        remove_package(step)
        pip(step, "install", str(wheel))
        merged, manifest, cell2, text = generate(step, [verify_code("local-path")], "local_path")
        merged_files.append(merged)
        expect(step, manifest.raw_installs == [], "a machine-specific path must not be stored", raw_installs=manifest.raw_installs)
        expect(step, str(wheel) not in text and str(scratch) not in text,
               "the local path must not appear anywhere in the generated notebook")
        expect(step, "system-dependent path" in cell2, "Cell 2 must say the package is on a system-dependent path", cell2=cell2[:1500])
        remove_package(step)
        stdout, errors = execute(step, merged)
        expect(step, len(errors) == 1 and errors[0].startswith("ModuleNotFoundError"),
               "with nothing to reinstall it, the import must fail visibly", errors=errors, stdout=stdout)
        print("   PASS")

    finally:
        server.shutdown()
        for merged in merged_files:
            if merged.exists():
                merged.unlink()
        run_cli_command([sys.executable, "-m", "pip", "uninstall", "-y", DIST_NAME])

    print("\nAll raw_installs end-to-end checks passed.")


if __name__ == "__main__":
    main()

"""Importing notebook_env must not touch process-global state; only the CLI entry point may.
Also pins that best-effort probes stay quiet by default but leave a trace under --verbose."""
import logging
import os
import subprocess
import sys
from pathlib import Path

import notebook_env as ne

MODULE_DIR = str(Path(ne.__file__).resolve().parent)


def _run(code: str, **env) -> str:
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=MODULE_DIR,
        env={**os.environ, **env},
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_import_leaves_streams_and_logging_untouched():
    out = _run(
        "import sys, logging\n"
        "before = sys.stdout.encoding\n"
        "import notebook_env\n"
        "log = logging.getLogger('notebook_env')\n"
        "print(sys.stdout.encoding == before, [type(h).__name__ for h in log.handlers], log.propagate)",
        PYTHONIOENCODING="latin-1",
    )
    assert out == "True ['NullHandler'] False"


def test_configure_console_is_where_streams_and_the_handler_get_set_up():
    """The stderr handler replaces the import-time placeholder, leaving exactly one handler."""
    out = _run(
        "import sys, logging, notebook_env\n"
        "notebook_env._configure_console()\n"
        "notebook_env._configure_console()  # idempotent\n"
        "log = logging.getLogger('notebook_env')\n"
        "print(sys.stdout.encoding, sorted(type(h).__name__ for h in log.handlers), log.propagate)",
        PYTHONIOENCODING="latin-1",
    )
    assert out == "utf-8 ['StreamHandler'] False"


def test_failed_opencv_probe_falls_back_and_is_logged_at_debug(monkeypatch, caplog):
    def boom(*args, **kwargs):
        raise OSError("pip is unavailable")

    monkeypatch.setattr(ne.subprocess, "run", boom)
    caplog.set_level(logging.DEBUG, logger="notebook_env")
    assert ne.resolve_opencv_variant() == "opencv-python"
    assert "Could not inspect installed OpenCV variants" in caplog.text


def test_reexecuting_the_source_in_one_process_never_stacks_handlers():
    """Pasting the tool into a live kernel more than once re-runs the whole file."""
    out = _run(
        "import logging, sys, notebook_env\n"
        "source = open(notebook_env.__file__, encoding='utf-8').read()\n"
        "for _ in range(2):\n"
        "    exec(compile(source, 'notebook_env.py', 'exec'), {'__name__': 'notebook_env_pasted'})\n"
        "    notebook_env._configure_console()\n"
        "print(len(logging.getLogger('notebook_env').handlers))"
    )
    assert out == "1"

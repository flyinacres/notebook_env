"""
End-to-end tests for manifest generation, extraction, tampering detection,
and check-drift -- against real files on disk and real PyPI (no mocking).

Companion to test_drift_check.py's fast mocked unit tests: this file is the
Docker/e2e tier, matching test_notebook_env_fixtures.py's own convention of
invoking the real pipeline rather than testing functions in isolation. Real
network calls are accepted here by design -- see the discussion that settled
this: e2e tests trade determinism for full-functionality coverage that unit
tests can't provide.

Assertions are chosen to be durable against live PyPI where possible (e.g.
requests==2.32.0's yank is a permanent historical fact, verified earlier
this session) to avoid the test breaking on schedule rather than on
regression.
"""

import json
from pathlib import Path

import pytest

import notebook_env as ne


FIXTURE_DIR = Path(__file__).parent / "fixtures"
KITCHEN_SINK_PATH = FIXTURE_DIR / "kitchen_sink.ipynb"


def _write_notebook_with_manifest(tmp_path, dependencies, filename="generated.ipynb"):
    """Builds a real .ipynb on disk with a genuinely-generated STEADY_PY_MANIFEST --
    via the actual generator, not hand-authored JSON."""
    result = ne.generate_production_blueprint(dependencies)
    nb = {
        "cells": [{
            "cell_type": "code", "source": [result["step2_code"]],
            "metadata": {}, "outputs": [], "execution_count": None,
        }],
        "metadata": {}, "nbformat": 4, "nbformat_minor": 5,
    }
    path = tmp_path / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f)
    return path, result


class TestManifestRoundTrip:
    def test_generate_then_extract_round_trip(self, tmp_path):
        """generate -> write -> extract should recover the exact same manifest."""
        deps = [{"name": "requests", "version": "2.32.1", "flags": []}]
        path, result = _write_notebook_with_manifest(tmp_path, deps)

        extracted, error = ne.extract_manifest_from_file(str(path))
        assert error is None
        assert extracted is not None

        original = result["drift_report"].manifest
        assert extracted.python_version == original.python_version
        assert extracted.dependencies == original.dependencies
        assert extracted.dependency_hash == original.dependency_hash
        assert extracted.tool_version == original.tool_version
        assert extracted.generated_at == original.generated_at

    def test_raw_installs_round_trip(self, tmp_path):
        """raw_installs (git/URL/local-path) must survive generate -> write -> extract intact."""
        result = ne.generate_production_blueprint([], raw_installs=["git+https://github.com/foo/bar.git@v1.2.0"])
        nb = {
            "cells": [{"cell_type": "code", "source": [result["step2_code"]],
                       "metadata": {}, "outputs": [], "execution_count": None}],
            "metadata": {}, "nbformat": 4, "nbformat_minor": 5,
        }
        path = tmp_path / "raw_install.ipynb"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f)

        extracted, error = ne.extract_manifest_from_file(str(path))
        assert error is None
        assert extracted.raw_installs == ["git+https://github.com/foo/bar.git@v1.2.0"]
        assert extracted.dependency_hash == result["drift_report"].manifest.dependency_hash

    def test_no_manifest_present_on_real_pre_feature_fixture(self):
        """kitchen_sink.ipynb predates this feature -- extraction must report
        'no manifest', not an error, against a real file that's never been
        through the generator."""
        if not KITCHEN_SINK_PATH.exists():
            pytest.fail(f"Fixture notebook not found at {KITCHEN_SINK_PATH}.")
        manifest, error = ne.extract_manifest_from_file(str(KITCHEN_SINK_PATH))
        assert manifest is None
        assert error is None

    def test_extraction_survives_real_shell_magic_lines(self, tmp_path):
        """Regression test for the magic-line parse bug: a notebook with a real
        '!pip install' line in an unrelated cell must not block extraction of
        a manifest that lives in a different cell."""
        deps = [{"name": "requests", "version": "2.32.1", "flags": []}]
        result = ne.generate_production_blueprint(deps)
        nb = {
            "cells": [
                {"cell_type": "code", "source": ["!pip install something-unrelated\n"],
                 "metadata": {}, "outputs": [], "execution_count": None},
                {"cell_type": "code", "source": [result["step2_code"]],
                 "metadata": {}, "outputs": [], "execution_count": None},
            ],
            "metadata": {}, "nbformat": 4, "nbformat_minor": 5,
        }
        path = tmp_path / "mixed_magic.ipynb"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f)

        manifest, error = ne.extract_manifest_from_file(str(path))
        assert error is None
        assert manifest is not None
        assert manifest.dependencies == deps


class TestTamperingDetectionE2E:
    def test_hand_edited_version_is_detected(self, tmp_path):
        """Real file, hand-edited on disk after generation -- hash mismatch fires."""
        deps = [{"name": "requests", "version": "2.32.1", "flags": []}]
        path, _ = _write_notebook_with_manifest(tmp_path, deps)

        content = path.read_text(encoding="utf-8")
        tampered = content.replace("'2.32.1'", "'2.32.0'")
        assert tampered != content, "replacement did not match anything in the generated file"
        path.write_text(tampered, encoding="utf-8")

        exit_code = ne.run_check_drift_pipeline(str(path))
        assert exit_code == 1  # confirmed findings present, no check_error

    def test_untampered_manifest_has_no_tampering_finding(self, tmp_path, capsys):
        deps = [{"name": "requests", "version": "2.32.1", "flags": []}]
        path, _ = _write_notebook_with_manifest(tmp_path, deps)

        ne.run_check_drift_pipeline(str(path))
        out = capsys.readouterr().out
        assert "[tampered]" not in out


class TestCheckDriftPipelineE2E:
    def test_yanked_pin_detected_via_real_file_and_real_pypi(self, tmp_path):
        """requests==2.32.0 is permanently yanked (verified live earlier this
        session) -- a durable fact, safe to assert against real PyPI without
        the test breaking as time passes."""
        deps = [{"name": "requests", "version": "2.32.0", "flags": []}]
        path, _ = _write_notebook_with_manifest(tmp_path, deps)

        exit_code = ne.run_check_drift_pipeline(str(path))
        assert exit_code == 1

    def test_no_manifest_exits_zero(self):
        exit_code = ne.run_check_drift_pipeline(str(KITCHEN_SINK_PATH))
        assert exit_code == 0

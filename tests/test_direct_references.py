"""
Packages installed from a direct reference (git URL, archive URL, local path,
editable install) rather than from PyPI.

Root cause these tests pin down: get_installed_environment assumed every
`pip freeze` line is `name==version`. Direct references (`name @ url`) were
dropped, and an editable install's comment line (`# Editable ... (name==0.1.0)`)
was parsed into a garbage entry, so an installed package was reported as
"not found".

Behavior under test:
  - Remote direct reference: not a PyPI pin; carried into raw_installs (unless
    the notebook already has that install line).
  - Local path / editable install: reported as system-dependent, never stored in
    the manifest, and the path never appears anywhere in the generated output.
"""
import json
import types

import pytest

import notebook_env as ne

REMOTE_URL = "git+https://example.com/org/zzq-remote.git@0123456789abcdef"
LOCAL_DIR = "/home/ron/src/zzq-local"

REMOTE_PIN = f"zzq-remote @ {REMOTE_URL}"
LOCAL_PIN = f"zzq-local @ file://{LOCAL_DIR}"

FREEZE_TEXT = "\n".join([
    "numpy==1.26.4",
    REMOTE_PIN,
    LOCAL_PIN,
    "# Editable Git install with no remote (zzq-editable==0.1.0)",
    "-e /home/ron/src/zzq-editable",
])


def _make_dist(site_dir, name, version, direct_url):
    info = site_dir / f"{name.replace('-', '_')}-{version}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    (info / "direct_url.json").write_text(json.dumps(direct_url))


@pytest.fixture
def fake_environment(tmp_path, monkeypatch):
    """A site dir holding direct-reference distributions, plus a canned `pip freeze`."""
    site = tmp_path / "site"
    site.mkdir()
    _make_dist(site, "zzq-remote", "1.0", {
        "url": "https://example.com/org/zzq-remote.git",
        "vcs_info": {"vcs": "git", "commit_id": "0123456789abcdef", "requested_revision": "v1.0"},
    })
    _make_dist(site, "zzq-local", "1.0", {"url": f"file://{LOCAL_DIR}", "dir_info": {}})
    _make_dist(site, "zzq-editable", "0.1.0", {
        "url": "file:///home/ron/src/zzq-editable", "dir_info": {"editable": True},
    })
    monkeypatch.syspath_prepend(str(site))
    monkeypatch.setattr(
        ne.subprocess, "run",
        lambda *a, **k: types.SimpleNamespace(returncode=0, stdout=FREEZE_TEXT, stderr=""),
    )
    return site


# --- get_installed_environment ------------------------------------------------

class TestInstalledEnvironmentParsing:
    def test_no_garbage_entries_from_comment_or_editable_lines(self, fake_environment):
        frozen, _ = ne.get_installed_environment()
        assert frozen["numpy"] == "numpy==1.26.4"
        bad = [k for k in frozen if k.startswith(("#", "-")) or " " in k or "install with no" in k]
        assert bad == []

    def test_remote_direct_reference_recorded_with_commit(self, fake_environment):
        frozen, _ = ne.get_installed_environment()
        assert frozen["zzq-remote"] == REMOTE_PIN

    def test_local_and_editable_recorded_as_direct_references(self, fake_environment):
        frozen, _ = ne.get_installed_environment()
        assert frozen["zzq-local"] == LOCAL_PIN
        assert frozen["zzq-editable"] == "zzq-editable @ file:///home/ron/src/zzq-editable"


# --- classification of a single import -----------------------------------------

class TestResolveDirectReference:
    def test_remote_is_not_a_pypi_pin_and_keeps_its_url(self):
        entry, _ = ne.resolve_pypi_package_and_extras(
            "zzqremote", set(), {"zzq-remote": REMOTE_PIN}, pkg_dist_map={"zzqremote": ["zzq-remote"]}
        )
        assert entry.status == "direct_reference"
        assert entry.is_comment is True
        assert entry.direct_url == REMOTE_URL
        assert "not found" not in entry.comment_text

    def test_local_is_system_path_and_never_carries_the_path(self):
        entry, _ = ne.resolve_pypi_package_and_extras(
            "zzqlocal", set(), {"zzq-local": LOCAL_PIN}, pkg_dist_map={"zzqlocal": ["zzq-local"]}
        )
        assert entry.status == "system_path"
        assert entry.is_comment is True
        assert entry.direct_url == ""
        assert LOCAL_DIR not in entry.comment_text
        assert "system-dependent path" in entry.comment_text

    def test_guarded_import_of_direct_reference_does_not_crash(self):
        entry, _ = ne.resolve_pypi_package_and_extras(
            "zzqremote", set(), {"zzq-remote": REMOTE_PIN},
            pkg_dist_map={"zzqremote": ["zzq-remote"]}, is_guarded=True,
        )
        assert entry.status == "guarded"
        assert REMOTE_URL not in entry.comment_text

    @pytest.mark.parametrize("builder,pkg", [
        (ne.build_auxiliary_tool_entries, "zzq-local"),
        (ne.build_writefile_tool_entries, "zzq_local"),
    ])
    def test_tool_entries_never_leak_a_local_path(self, builder, pkg):
        entries = builder({pkg}, set(), {"zzq-local": LOCAL_PIN})
        assert entries
        assert LOCAL_DIR not in " ".join(e.comment_text for e in entries)


# --- same-source comparison (dedupe against the notebook's own install line) ----

class TestSameDirectSource:
    @pytest.mark.parametrize("a,b", [
        ("git+https://example.com/org/x.git@v1.0", "git+https://example.com/org/x.git@0123abc"),
        ("git+https://example.com/org/x.git@v1.0", "git+https://example.com/org/x"),
        ("git+https://example.com/org/x.git#egg=x", "git+https://example.com/org/x.git@abc"),
        ("git+https://user@example.com/org/x.git@v1", "git+https://user@example.com/org/x.git@abc"),
        ("GIT+HTTPS://EXAMPLE.COM/org/x.git", "git+https://example.com/org/x.git@abc"),
    ])
    def test_same_source_different_ref(self, a, b):
        assert ne.same_direct_source(a, b)

    @pytest.mark.parametrize("a,b", [
        ("git+https://example.com/org/x.git@v1", "git+https://example.com/org/y.git@v1"),
        ("git+https://example.com/org/x.git", "git+https://other.example.com/org/x.git"),
    ])
    def test_different_source(self, a, b):
        assert not ne.same_direct_source(a, b)


# --- end to end through the generator -------------------------------------------

def _generate(tmp_path, code_sources, imports, frozen_env, pkg_dist_map, raw_installs=None):
    nb_path = tmp_path / "nb.ipynb"
    nb_path.write_text(json.dumps({
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                   "source": [src]} for src in code_sources],
        "metadata": {"kernelspec": {"language": "python", "name": "python3", "display_name": "Python 3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }))
    scan_res = ne.NotebookScanResult(
        path=nb_path, is_python=True, lang_label="python",
        imports=imports, code_sources=code_sources,
        raw_installs=list(raw_installs or []),
    )
    out_path, _ = ne.apply_output_to_notebook(scan_res, frozen_env, pkg_dist_map, None, suffix="_out")
    manifest, error = ne.extract_manifest_from_file(str(out_path))
    assert error is None
    return manifest, out_path.read_text()


class TestGeneratedManifest:
    def test_remote_reference_is_carried_into_raw_installs(self, tmp_path):
        manifest, _ = _generate(
            tmp_path, ["import zzqremote\n"], {"zzqremote"},
            {"zzq-remote": REMOTE_PIN}, {"zzqremote": ["zzq-remote"]},
        )
        assert manifest.raw_installs == [REMOTE_URL]
        assert all("zzq-remote" not in d["name"] for d in manifest.dependencies)

    def test_notebooks_own_install_line_is_not_duplicated(self, tmp_path):
        author_spec = "git+https://example.com/org/zzq-remote.git@v1.0"
        manifest, _ = _generate(
            tmp_path, [f"%pip install {author_spec}\n", "import zzqremote\n"], {"zzqremote"},
            {"zzq-remote": REMOTE_PIN}, {"zzqremote": ["zzq-remote"]},
            raw_installs=[author_spec],
        )
        assert manifest.raw_installs == [author_spec]

    def test_local_path_is_not_stored_and_never_appears_in_output(self, tmp_path):
        manifest, text = _generate(
            tmp_path, ["import zzqlocal\n"], {"zzqlocal"},
            {"zzq-local": LOCAL_PIN}, {"zzqlocal": ["zzq-local"]},
        )
        assert manifest.raw_installs == []
        assert manifest.dependencies == []
        assert LOCAL_DIR not in text
        assert "system-dependent path" in text

    def test_guarded_direct_reference_adds_nothing_to_the_manifest(self, tmp_path):
        manifest, _ = _generate(
            tmp_path, ["try:\n    import zzqremote\nexcept ImportError:\n    pass\n"], {"zzqremote"},
            {"zzq-remote": REMOTE_PIN}, {"zzqremote": ["zzq-remote"]},
        )
        assert manifest.raw_installs == []

"""
Unit tests for pinned-dependency drift detection.

Mocking boundary is `_fetch_pypi_json` (the single HTTP choke point both
fetch_pypi_version_metadata and fetch_pypi_package_metadata call through) --
no real network calls in this suite, but the actual JSON-parsing logic in
both fetch functions still runs and is exercised, not bypassed.
"""

import pytest

import notebook_env as ne


# ---------------------------------------------------------------------------
# Fake PyPI fixture data + fetcher
# ---------------------------------------------------------------------------

FAKE_PACKAGES = {
    "pandas": {
        "latest_version": "2.2.1",
        "releases": {
            "2.2.1": {"upload_time": "2024-02-23T00:00:00.000000Z", "yanked": False},
        },
        "versions": {
            "2.2.1": {
                "requires_dist": [
                    'numpy<2,>=1.22.4; python_version < "3.11"',
                    'numpy<2,>=1.23.2; python_version == "3.11"',
                    'numpy<2,>=1.26.0; python_version >= "3.12"',
                    'hypothesis>=6.46.1; extra == "test"',
                ],
                "requires_python": ">=3.9",
                "yanked": False,
                "yanked_reason": None,
                "project_urls": {},
            },
        },
    },
    "numpy": {
        "latest_version": "2.5.3",
        "releases": {
            "1.26.4": {"upload_time": "2024-02-06T00:00:00.000000Z", "yanked": False},
            "2.5.3": {"upload_time": "2026-08-01T00:00:00.000000Z", "yanked": False},
        },
        "versions": {
            "1.26.4": {
                "requires_dist": [], "requires_python": ">=3.9", "yanked": False,
                "yanked_reason": None, "project_urls": {},
            },
            "2.5.3": {
                "requires_dist": [], "requires_python": ">=3.11", "yanked": False,
                "yanked_reason": None, "project_urls": {},
            },
        },
    },
    "requests": {
        "latest_version": "2.32.1",
        "releases": {
            "2.32.0": {"upload_time": "2024-05-20T00:00:00.000000Z", "yanked": True},
            "2.32.1": {"upload_time": "2024-05-21T00:00:00.000000Z", "yanked": False},
        },
        "versions": {
            "2.32.0": {
                "requires_dist": [], "requires_python": None, "yanked": True,
                "yanked_reason": "CVE-2024-35195 mitigation conflict", "project_urls": {},
            },
        },
    },
    "old-package": {
        # exists, but its only version is missing from "versions" -> version-removed case
        "latest_version": "1.0.0",
        "releases": {
            "1.0.0": {"upload_time": "2018-01-01T00:00:00.000000Z", "yanked": False},
        },
        "versions": {},  # 0.9.0 (tested below) deliberately absent
    },
    "stale-package": {
        "latest_version": "1.0.0",
        "releases": {
            "1.0.0": {"upload_time": "2018-01-01T00:00:00.000000Z", "yanked": False},
        },
        "versions": {
            "1.0.0": {
                "requires_dist": [], "requires_python": None, "yanked": False,
                "yanked_reason": None, "project_urls": {},
            },
        },
    },
    "flaky-package": {
        "network_error": True,  # every lookup for this package simulates a network failure
    },
}


def make_fake_fetch(packages):
    def _fake_fetch(url):
        path = url[len("https://pypi.org/pypi/"):].rstrip("/")
        if path.endswith("/json"):
            path = path[: -len("/json")]
        parts = path.split("/")

        if len(parts) == 2:
            name, version = parts
        else:
            name, version = parts[0], None

        pkg = packages.get(name)
        if pkg is None:
            return "not_found", None, None
        if pkg.get("network_error"):
            return "network_error", None, "simulated connection failure"

        if version is not None:
            ver_data = pkg["versions"].get(version)
            if ver_data is None:
                return "not_found", None, None
            payload = {"info": {
                "requires_dist": ver_data.get("requires_dist", []),
                "requires_python": ver_data.get("requires_python"),
                "yanked": ver_data.get("yanked", False),
                "yanked_reason": ver_data.get("yanked_reason"),
                "project_urls": ver_data.get("project_urls", {}),
            }}
            return "found", payload, None

        releases_payload = {}
        for ver, info in pkg["releases"].items():
            releases_payload[ver] = [{
                "upload_time_iso_8601": info["upload_time"],
                "yanked": info["yanked"],
            }]
        payload = {"info": {"version": pkg["latest_version"]}, "releases": releases_payload}
        return "found", payload, None

    return _fake_fetch


@pytest.fixture(autouse=True)
def _mock_pypi(monkeypatch):
    """Applies to every test in this module: no real network calls, and the
    memoization caches never leak state between tests."""
    monkeypatch.setattr(ne, "_fetch_pypi_json", make_fake_fetch(FAKE_PACKAGES))
    ne.fetch_pypi_version_metadata.cache_clear()
    ne.fetch_pypi_package_metadata.cache_clear()
    yield
    ne.fetch_pypi_version_metadata.cache_clear()
    ne.fetch_pypi_package_metadata.cache_clear()


REQ_PY_311 = {"major": 3, "minor": 11}


# ---------------------------------------------------------------------------
# PyPI metadata client
# ---------------------------------------------------------------------------

class TestPypiVersionMetadata:
    def test_found(self):
        meta = ne.fetch_pypi_version_metadata("pandas", "2.2.1")
        assert meta.status == "found"
        assert meta.requires_python == ">=3.9"
        assert meta.yanked is False
        assert len(meta.requires_dist) == 4

    def test_yanked_with_reason(self):
        meta = ne.fetch_pypi_version_metadata("requests", "2.32.0")
        assert meta.status == "found"
        assert meta.yanked is True
        assert meta.yanked_reason == "CVE-2024-35195 mitigation conflict"

    def test_version_not_found(self):
        meta = ne.fetch_pypi_version_metadata("old-package", "0.9.0")
        assert meta.status == "not_found"

    def test_whole_package_not_found(self):
        meta = ne.fetch_pypi_version_metadata("fake-package-xyz", "1.0.0")
        assert meta.status == "not_found"

    def test_network_error_is_distinct_from_not_found(self):
        meta = ne.fetch_pypi_version_metadata("flaky-package", "1.0.0")
        assert meta.status == "network_error"
        assert meta.error_detail

    def test_memoized_within_run(self, monkeypatch):
        calls = []
        real_fetch = ne._fetch_pypi_json
        def counting_fetch(url):
            calls.append(url)
            return real_fetch(url)
        monkeypatch.setattr(ne, "_fetch_pypi_json", counting_fetch)

        ne.fetch_pypi_version_metadata("pandas", "2.2.1")
        ne.fetch_pypi_version_metadata("pandas", "2.2.1")
        assert len(calls) == 1


class TestPypiPackageMetadata:
    def test_found(self):
        meta = ne.fetch_pypi_package_metadata("numpy")
        assert meta.status == "found"
        assert meta.latest_version == "2.5.3"
        assert meta.releases["1.26.4"]["yanked"] is False

    def test_yanked_release_surfaces_in_releases_dict(self):
        meta = ne.fetch_pypi_package_metadata("requests")
        assert meta.releases["2.32.0"]["yanked"] is True

    def test_not_found(self):
        meta = ne.fetch_pypi_package_metadata("fake-package-xyz")
        assert meta.status == "not_found"

    def test_network_error(self):
        meta = ne.fetch_pypi_package_metadata("flaky-package")
        assert meta.status == "network_error"


# ---------------------------------------------------------------------------
# Direct-pin checks
# ---------------------------------------------------------------------------

class TestYankedOrRemoved:
    def test_clean_pin_no_finding(self):
        assert ne.check_yanked_or_removed("pandas", "2.2.1") == []

    def test_yanked(self):
        findings = ne.check_yanked_or_removed("requests", "2.32.0")
        assert len(findings) == 1
        assert findings[0].signal == "yanked"
        assert findings[0].severity == "confirmed"

    def test_version_removed_project_alive(self):
        findings = ne.check_yanked_or_removed("old-package", "0.9.0")
        assert len(findings) == 1
        assert findings[0].signal == "removed"
        assert "still published" in findings[0].message

    def test_whole_project_never_found_on_pypi(self):
        """Neutral wording: never presumes the package once existed (it may never have)."""
        findings = ne.check_yanked_or_removed("fake-package-xyz", "1.0.0")
        assert len(findings) == 1
        assert findings[0].signal == "not_found_on_pypi"
        assert findings[0].severity == "confirmed"
        assert "could not be found on PyPI" in findings[0].message

    def test_pip_env_hint_appended_when_set(self, monkeypatch):
        monkeypatch.setenv("PIP_FIND_LINKS", "/some/local/dist")
        findings = ne.check_yanked_or_removed("fake-package-xyz", "1.0.0")
        assert "PIP_FIND_LINKS=/some/local/dist" in findings[0].message

    def test_pip_env_hint_absent_when_not_set(self, monkeypatch):
        for var in ("PIP_FIND_LINKS", "PIP_NO_INDEX", "PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL"):
            monkeypatch.delenv(var, raising=False)
        findings = ne.check_yanked_or_removed("fake-package-xyz", "1.0.0")
        assert "Note:" not in findings[0].message

    def test_network_error_reported_not_silenced(self):
        findings = ne.check_yanked_or_removed("flaky-package", "1.0.0")
        assert len(findings) == 1
        assert findings[0].signal == "check_error"
        assert findings[0].severity == "error"

    def test_extras_tag_stripped_before_lookup(self):
        """Regression test: a pin name carrying an extras tag (e.g. from extras
        promotion) must not be treated as a literal PyPI project name."""
        findings = ne.check_yanked_or_removed("pandas[test]", "2.2.1")
        assert findings == []


class TestStaleness:
    def test_active_package_no_finding(self):
        assert ne.check_staleness("numpy", "2.5.3") == []

    def test_stale_package_flagged_as_heuristic(self):
        findings = ne.check_staleness("stale-package", "1.0.0")
        assert len(findings) == 1
        assert findings[0].signal == "stale"
        assert findings[0].severity == "heuristic"

    def test_network_error_reported_not_silenced(self):
        findings = ne.check_staleness("flaky-package", "1.0.0")
        assert len(findings) == 1
        assert findings[0].signal == "check_error"
        assert findings[0].severity == "error"


class TestMajorBump:
    def test_no_bump_available(self):
        assert ne.check_major_bump("numpy", "2.5.3") == []

    def test_bump_available_is_heuristic(self):
        findings = ne.check_major_bump("numpy", "1.26.4")
        assert len(findings) == 1
        assert findings[0].signal == "major_bump"
        assert findings[0].severity == "heuristic"
        assert findings[0].details["latest_version"] == "2.5.3"

    def test_network_error_reported_not_silenced(self):
        findings = ne.check_major_bump("flaky-package", "1.0.0")
        assert len(findings) == 1
        assert findings[0].signal == "check_error"
        assert findings[0].severity == "error"


class TestPythonSupport:
    def test_supported(self):
        assert ne.check_python_support("pandas", "2.2.1", {"major": 3, "minor": 11}) == []

    def test_unsupported(self):
        findings = ne.check_python_support("numpy", "2.5.3", {"major": 3, "minor": 8})
        assert len(findings) == 1
        assert findings[0].signal == "unsupported_python"
        assert findings[0].severity == "confirmed"

    def test_no_requires_python_declared_is_not_a_finding(self):
        assert ne.check_python_support("stale-package", "1.0.0", {"major": 3, "minor": 8}) == []

    def test_network_error_reported_not_silenced(self):
        findings = ne.check_python_support("flaky-package", "1.0.0", {"major": 3, "minor": 11})
        assert len(findings) == 1
        assert findings[0].signal == "check_error"
        assert findings[0].severity == "error"


# ---------------------------------------------------------------------------
# Marker evaluation
# ---------------------------------------------------------------------------

class TestMarkerEnvironment:
    def test_extra_marker_false_for_base_install(self):
        env = ne._marker_environment(REQ_PY_311, extra=None)
        assert env["extra"] == ""

    def test_python_version_marker_uses_required_python(self):
        env = ne._marker_environment({"major": 3, "minor": 9}, extra=None)
        assert env["python_version"] == "3.9"

    def test_split_pin_name_extracts_extra(self):
        name, extra = ne._split_pin_name("pandas[test]")
        assert name == "pandas"
        assert extra == "test"

    def test_split_pin_name_no_extra(self):
        name, extra = ne._split_pin_name("pandas")
        assert name == "pandas"
        assert extra is None


# ---------------------------------------------------------------------------
# Transitive resolution
# ---------------------------------------------------------------------------

class TestResolveTransitiveGraph:
    def test_resolvable_graph(self):
        deps = [{"name": "pandas", "version": "2.2.1", "flags": []}]
        resolved, findings = ne.resolve_transitive_graph(deps, REQ_PY_311)
        assert findings == []
        assert resolved["pandas"] == "2.2.1"
        assert resolved["numpy"] == "1.26.4"  # only numpy version satisfying pandas's 3.11 branch

    def test_unresolvable_graph_reports_conflict(self):
        deps = [
            {"name": "pandas", "version": "2.2.1", "flags": []},
            {"name": "numpy", "version": "2.5.3", "flags": []},
        ]
        resolved, findings = ne.resolve_transitive_graph(deps, REQ_PY_311)
        assert resolved is None
        assert len(findings) >= 1
        assert all(f.signal == "conflict" and f.severity == "confirmed" for f in findings)

    def test_extra_gated_requirement_excluded_from_base_resolution(self):
        """pandas's hypothesis requirement is extra=='test'-gated; a base install
        (no extras requested) must not pull it into the graph."""
        deps = [{"name": "pandas", "version": "2.2.1", "flags": []}]
        resolved, findings = ne.resolve_transitive_graph(deps, REQ_PY_311)
        assert "hypothesis" not in resolved


class TestCheckTransitiveSignals:
    def test_direct_pins_are_skipped(self):
        deps = [{"name": "pandas", "version": "2.2.1", "flags": []}]
        findings = ne.check_transitive_signals(deps, REQ_PY_311)
        assert all(f.package != "pandas" for f in findings)

    def test_transitive_package_checked_against_resolved_version(self):
        deps = [{"name": "pandas", "version": "2.2.1", "flags": []}]
        findings = ne.check_transitive_signals(deps, {"major": 3, "minor": 8})
        # numpy resolves to 1.26.4 here (only version satisfying pandas's non-3.11/3.12 branch);
        # 1.26.4 declares requires-python >=3.9, which doesn't cover 3.8.
        unsupported = [f for f in findings if f.signal == "unsupported_python" and f.package == "numpy"]
        assert len(unsupported) == 1

    def test_unresolvable_graph_short_circuits_to_conflict_findings(self):
        deps = [
            {"name": "pandas", "version": "2.2.1", "flags": []},
            {"name": "numpy", "version": "2.5.3", "flags": []},
        ]
        findings = ne.check_transitive_signals(deps, REQ_PY_311)
        assert all(f.signal == "conflict" for f in findings)


class TestLocalModuleDriftCheck:
    """
    Unit coverage for check_local_modules -- pure filesystem existence checks,
    no PyPI/network involved. Covers all real outcomes: still found (no
    finding), genuinely gone (confirmed), and unverifiable (error) in both
    ways that can happen -- no root_dir supplied, or the anchor directory
    itself no longer exists.
    """

    def _manifest_with(self, local_modules):
        return ne.SteadyPyManifest(
            python_version={"major": 3, "minor": 11},
            dependencies=[],
            gpu=None,
            generated_at="2026-01-01 00:00:00",
            local_modules=local_modules,
        )

    def test_notebook_dir_module_still_present_no_finding(self, tmp_path):
        (tmp_path / "cookbook.py").write_text("# helper", encoding="utf-8")
        manifest = self._manifest_with([{"name": "cookbook", "anchor": "notebook_dir"}])

        findings = ne.check_local_modules(manifest, notebook_dir=str(tmp_path))
        assert findings == []

    def test_notebook_dir_module_missing_is_confirmed(self, tmp_path):
        """Module never existed at this path (or was removed) -- notebook_dir
        itself always exists here since we're checking against a real tmp_path,
        so this must land as a genuine confirmed finding, not unverifiable."""
        manifest = self._manifest_with([{"name": "cookbook", "anchor": "notebook_dir"}])

        findings = ne.check_local_modules(manifest, notebook_dir=str(tmp_path))
        assert len(findings) == 1
        assert findings[0].signal == "local_module_missing"
        assert findings[0].severity == "confirmed"
        assert "cookbook" in findings[0].message

    def test_root_dir_module_not_supplied_is_unverifiable(self, tmp_path):
        manifest = self._manifest_with([{"name": "shared_utils", "anchor": "root_dir"}])

        findings = ne.check_local_modules(manifest, notebook_dir=str(tmp_path), root_dir=None)
        assert len(findings) == 1
        assert findings[0].signal == "local_module_unverifiable"
        assert findings[0].severity == "error"
        assert "none was supplied" in findings[0].message

    def test_root_dir_module_still_present_no_finding(self, tmp_path):
        (tmp_path / "shared_utils.py").write_text("# shared", encoding="utf-8")
        manifest = self._manifest_with([{"name": "shared_utils", "anchor": "root_dir"}])

        findings = ne.check_local_modules(manifest, notebook_dir=str(tmp_path / "nb_dir"), root_dir=str(tmp_path))
        assert findings == []

    def test_root_dir_module_deleted_is_confirmed(self, tmp_path):
        manifest = self._manifest_with([{"name": "shared_utils", "anchor": "root_dir"}])

        findings = ne.check_local_modules(manifest, notebook_dir=str(tmp_path / "nb_dir"), root_dir=str(tmp_path))
        assert len(findings) == 1
        assert findings[0].signal == "local_module_missing"
        assert findings[0].severity == "confirmed"

    def test_root_dir_itself_gone_is_unverifiable_not_confirmed(self, tmp_path):
        """Distinguishes 'the whole project moved' from 'this one file is gone' --
        must not be reported as a confirmed missing module, since the module
        may well still exist at wherever the project moved to."""
        missing_root = str(tmp_path / "does_not_exist")
        manifest = self._manifest_with([{"name": "shared_utils", "anchor": "root_dir"}])

        findings = ne.check_local_modules(manifest, notebook_dir=str(tmp_path), root_dir=missing_root)
        assert len(findings) == 1
        assert findings[0].signal == "local_module_unverifiable"
        assert findings[0].severity == "error"
        assert "no longer exists" in findings[0].message

    def test_multiple_entries_each_get_independent_findings(self, tmp_path):
        (tmp_path / "present.py").write_text("# present", encoding="utf-8")
        manifest = self._manifest_with([
            {"name": "present", "anchor": "notebook_dir"},
            {"name": "absent", "anchor": "notebook_dir"},
        ])

        findings = ne.check_local_modules(manifest, notebook_dir=str(tmp_path))
        assert len(findings) == 1
        assert findings[0].package == "absent"

    def test_entry_missing_name_key_is_skipped_not_crashed(self, tmp_path):
        manifest = self._manifest_with([{"anchor": "notebook_dir"}])
        findings = ne.check_local_modules(manifest, notebook_dir=str(tmp_path))
        assert findings == []
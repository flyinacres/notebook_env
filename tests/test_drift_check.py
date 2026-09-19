"""
Unit tests for pinned-dependency drift detection.

Mocking boundary is `_fetch_pypi_json` (the single HTTP choke point both
fetch_pypi_version_metadata and fetch_pypi_package_metadata call through) --
no real network calls in this suite, but the actual JSON-parsing logic in
both fetch functions still runs and is exercised, not bypassed.
"""

import hashlib
import json

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


def _pkg(version="1.0.0", requires_dist=(), yanked=False):
    """One-release fake package, uploaded recently so it never trips the staleness heuristic."""
    return {
        "latest_version": version,
        "releases": {version: {"upload_time": "2026-08-01T00:00:00.000000Z", "yanked": yanked}},
        "versions": {version: {
            "requires_dist": list(requires_dist), "requires_python": None, "yanked": yanked,
            "yanked_reason": "fixture: yanked" if yanked else None, "project_urls": {},
        }},
    }


# pandas[test] -> hypothesis (>=6.46.1) -> sortedcontainers, which is only reachable through the extra.
FAKE_PACKAGES["hypothesis"] = {
    "latest_version": "6.100.0",
    "releases": {
        "5.0.0": {"upload_time": "2026-08-01T00:00:00.000000Z", "yanked": False},
        "6.100.0": {"upload_time": "2026-08-01T00:00:00.000000Z", "yanked": False},
    },
    "versions": {
        "5.0.0": {"requires_dist": [], "requires_python": None, "yanked": False,
                  "yanked_reason": None, "project_urls": {}},
        "6.100.0": {"requires_dist": ["sortedcontainers>=2.1.0"], "requires_python": None, "yanked": False,
                    "yanked_reason": None, "project_urls": {}},
    },
}
FAKE_PACKAGES["sortedcontainers"] = _pkg("2.4.0", yanked=True)

# Two independent extras plus an unconditional requirement.
FAKE_PACKAGES["multi-extra-pkg"] = _pkg("1.0.0", requires_dist=[
    'alpha-dep; extra == "a"',
    'beta-dep; extra == "b"',
    "core-dep",
])
FAKE_PACKAGES["alpha-dep"] = _pkg()
FAKE_PACKAGES["beta-dep"] = _pkg()
FAKE_PACKAGES["core-dep"] = _pkg()

# A package whose own requirement asks for an extra of another package.
FAKE_PACKAGES["meta-pkg"] = _pkg("1.0.0", requires_dist=["pandas[test]>=2.2"])


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


class TestExtrasInTransitiveGraph:
    """A pin like pandas[test] must pull the extra's own requirements into the graph."""

    def test_extra_requirements_are_walked(self):
        deps = [{"name": "pandas[test]", "version": "2.2.1", "flags": []}]
        resolved, findings = ne.resolve_transitive_graph(deps, REQ_PY_311)
        assert findings == []
        assert resolved["hypothesis"] == "6.100.0"
        assert "sortedcontainers" in resolved  # reachable only through the extra

    def test_base_pin_still_excludes_extra_requirements(self):
        deps = [{"name": "pandas", "version": "2.2.1", "flags": []}]
        resolved, _ = ne.resolve_transitive_graph(deps, REQ_PY_311)
        assert "hypothesis" not in resolved

    def test_extras_variant_is_not_reported_as_a_separate_package(self):
        deps = [{"name": "pandas[test]", "version": "2.2.1", "flags": []}]
        resolved, _ = ne.resolve_transitive_graph(deps, REQ_PY_311)
        assert not [name for name in resolved if "[" in name]
        assert resolved["pandas"] == "2.2.1"

    def test_every_requested_extra_is_walked(self):
        deps = [{"name": "multi-extra-pkg[a,b]", "version": "1.0.0", "flags": []}]
        resolved, findings = ne.resolve_transitive_graph(deps, REQ_PY_311)
        assert findings == []
        assert {"alpha-dep", "beta-dep", "core-dep"} <= set(resolved)

    def test_only_the_requested_extra_is_walked(self):
        deps = [{"name": "multi-extra-pkg[a]", "version": "1.0.0", "flags": []}]
        resolved, _ = ne.resolve_transitive_graph(deps, REQ_PY_311)
        assert "alpha-dep" in resolved
        assert "beta-dep" not in resolved
        assert "core-dep" in resolved

    def test_extra_named_by_a_transitive_requirement_is_walked(self):
        deps = [{"name": "meta-pkg", "version": "1.0.0", "flags": []}]
        resolved, findings = ne.resolve_transitive_graph(deps, REQ_PY_311)
        assert findings == []
        assert "hypothesis" in resolved  # meta-pkg -> pandas[test] -> hypothesis

    def test_conflict_created_by_an_extra_is_reported(self):
        deps = [
            {"name": "pandas[test]", "version": "2.2.1", "flags": []},
            {"name": "hypothesis", "version": "5.0.0", "flags": []},  # pandas[test] needs >=6.46.1
        ]
        resolved, findings = ne.resolve_transitive_graph(deps, REQ_PY_311)
        assert resolved is None
        assert findings and all(f.signal == "conflict" and f.severity == "confirmed" for f in findings)

    def test_unknown_extra_adds_nothing_and_does_not_fail(self):
        deps = [{"name": "pandas[nonexistent]", "version": "2.2.1", "flags": []}]
        resolved, findings = ne.resolve_transitive_graph(deps, REQ_PY_311)
        assert findings == []
        assert "hypothesis" not in resolved
        assert resolved["pandas"] == "2.2.1"

    def test_signals_reach_packages_only_reachable_through_the_extra(self):
        with_extra = ne.check_transitive_signals(
            [{"name": "pandas[test]", "version": "2.2.1", "flags": []}], REQ_PY_311)
        assert [f for f in with_extra if f.signal == "yanked" and f.package == "sortedcontainers"]

        without = ne.check_transitive_signals(
            [{"name": "pandas", "version": "2.2.1", "flags": []}], REQ_PY_311)
        assert not [f for f in without if f.package == "sortedcontainers"]

    def test_all_requested_extras_are_parsed(self):
        name, extras = ne._split_pin_extras("multi-extra-pkg[b,a]")
        assert name == "multi-extra-pkg"
        assert extras == frozenset({"a", "b"})


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

# ---------------------------------------------------------------------------
# Manifest hash verification, shared pin checks, generation-time ordering
# ---------------------------------------------------------------------------

CLEAN_DEP = {"name": "core-dep", "version": "1.0.0", "flags": []}  # fake package with no findings of any kind


def _sha256_of(payload):
    """Independent of the implementation: canonical JSON of everything except dependency_hash."""
    body = {k: v for k, v in payload.items() if k != "dependency_hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _old_style_manifest():
    """Shaped and hashed the way a tool version that predates the local_modules field wrote it."""
    manifest = {
        "python_version": {"major": 3, "minor": 11},
        "dependencies": [dict(CLEAN_DEP)],
        "gpu": None,
        "generated_at": "2025-01-01 00:00:00",
        "tool_version": "40",
        "raw_installs": [],
        "custom_sourced": [],
    }
    manifest["dependency_hash"] = _sha256_of(manifest)
    return manifest


def _write_literal(tmp_path, manifest, name="nb.py"):
    path = tmp_path / name
    path.write_text(f"STEADY_PY_MANIFEST = {manifest!r}\n", encoding="utf-8")
    return path


def _check(path, capsys):
    exit_code = ne.run_check_drift_pipeline(str(path), output_format="json")
    return exit_code, json.loads(capsys.readouterr().out)


class TestManifestHashVerification:
    """The hash is verified over the data exactly as persisted, not over the current dataclass shape."""

    def test_manifest_from_before_a_field_existed_still_verifies(self, tmp_path, capsys):
        exit_code, report = _check(_write_literal(tmp_path, _old_style_manifest()), capsys)
        assert [f for f in report["confirmed"] if f["signal"] == "tampered"] == []
        assert exit_code == 0

    def test_freshly_generated_manifest_verifies(self, tmp_path, capsys):
        result = ne.generate_production_blueprint([dict(CLEAN_DEP)])
        exit_code, report = _check(_write_literal(tmp_path, result["drift_report"].manifest.to_dict()), capsys)
        assert [f for f in report["confirmed"] if f["signal"] == "tampered"] == []
        assert exit_code == 0

    def test_hand_edited_value_is_detected(self, tmp_path, capsys):
        manifest = _old_style_manifest()
        manifest["generated_at"] = "2026-09-01 00:00:00"  # hiding age, hash left alone
        exit_code, report = _check(_write_literal(tmp_path, manifest), capsys)
        assert [f for f in report["confirmed"] if f["signal"] == "tampered"]
        assert exit_code == 1

    def test_deleting_a_field_is_detected(self, tmp_path, capsys):
        result = ne.generate_production_blueprint([dict(CLEAN_DEP)])
        manifest = result["drift_report"].manifest.to_dict()
        del manifest["local_modules"]
        _, report = _check(_write_literal(tmp_path, manifest), capsys)
        assert [f for f in report["confirmed"] if f["signal"] == "tampered"]

    def test_report_carries_the_stored_hash_not_the_recomputed_one(self, tmp_path, capsys):
        manifest = _old_style_manifest()
        stored = manifest["dependency_hash"]
        manifest["generated_at"] = "2026-09-01 00:00:00"
        _, report = _check(_write_literal(tmp_path, manifest), capsys)
        assert report["manifest"]["dependency_hash"] == stored
        tampered = [f for f in report["confirmed"] if f["signal"] == "tampered"][0]
        assert tampered["details"]["stored_hash"] == stored
        assert tampered["details"]["recomputed_hash"] != stored


_PIN_CHECKS = [
    "check_yanked_or_removed", "check_staleness", "check_major_bump",
    "check_python_support", "check_transitive_signals",
]


class TestPinChecksAreSharedBetweenGenerationAndCheckDrift:
    """Both moments must run the very same checks on the very same inputs. They were once two
    hand-maintained copies of one sequence; this guards against that drifting apart again."""

    def _record_calls(self, monkeypatch, log):
        for fname in _PIN_CHECKS:
            monkeypatch.setattr(ne, fname, lambda *args, _f=fname: log.append((_f, args)) or [])

    def test_identical_checks_in_both_paths(self, tmp_path, monkeypatch, capsys):
        deps = [
            dict(CLEAN_DEP),
            {"name": "torch", "version": "2.3.1+cu121", "flags": []},  # local version: skipped by direct checks
            {"name": "pandas[test]", "version": "2.2.1", "flags": []},
        ]
        generated, checked = [], []

        self._record_calls(monkeypatch, generated)
        result = ne.generate_production_blueprint(deps)
        path = _write_literal(tmp_path, result["drift_report"].manifest.to_dict())

        self._record_calls(monkeypatch, checked)
        ne.run_check_drift_pipeline(str(path), output_format="json")
        capsys.readouterr()

        assert generated, "the generation path ran no pin checks at all"
        assert generated == checked

    def test_local_version_pin_yields_the_same_finding_in_both_paths(self, tmp_path, capsys):
        deps = [{"name": "torch", "version": "2.3.1+cu121", "flags": []}]
        result = ne.generate_production_blueprint(deps)
        at_generation = [f.to_dict() for f in result["drift_report"].heuristic]
        path = _write_literal(tmp_path, result["drift_report"].manifest.to_dict())
        _, report = _check(path, capsys)

        assert [f["signal"] for f in at_generation] == ["unverifiable_custom_index"]
        assert report["heuristic"] == at_generation


class TestGenerationOrdering:
    def test_pin_checks_run_before_the_manifest_is_hashed(self, monkeypatch):
        """Recorded findings will live inside the hashed manifest, so they must exist first."""
        order = []
        monkeypatch.setattr(ne, "check_yanked_or_removed", lambda *a: order.append("checks") or [])
        real_hash = ne.SteadyPyManifest.compute_and_set_hash

        def spy(self):
            order.append("hash")
            return real_hash(self)

        monkeypatch.setattr(ne.SteadyPyManifest, "compute_and_set_hash", spy)
        ne.generate_production_blueprint([dict(CLEAN_DEP)])
        assert "checks" in order and "hash" in order
        assert order.index("checks") < order.index("hash")
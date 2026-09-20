"""
Unit tests for pinned-dependency drift detection.

Mocking boundary is `_fetch_pypi_json` (the single HTTP choke point both
fetch_pypi_version_metadata and fetch_pypi_package_metadata call through) --
no real network calls in this suite, but the actual JSON-parsing logic in
both fetch functions still runs and is exercised, not bypassed.
"""

import argparse
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
        assert findings[0].latest_version == "2.5.3"

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
        deps = [ne.PinnedDependency("pandas", "2.2.1")]
        resolved, findings = ne.resolve_transitive_graph(deps, REQ_PY_311)
        assert findings == []
        assert resolved["pandas"] == "2.2.1"
        assert resolved["numpy"] == "1.26.4"  # only numpy version satisfying pandas's 3.11 branch

    def test_unresolvable_graph_reports_conflict(self):
        deps = [
            ne.PinnedDependency("pandas", "2.2.1"),
            ne.PinnedDependency("numpy", "2.5.3"),
        ]
        resolved, findings = ne.resolve_transitive_graph(deps, REQ_PY_311)
        assert resolved is None
        assert len(findings) >= 1
        assert all(f.signal == "conflict" and f.severity == "confirmed" for f in findings)

    def test_extra_gated_requirement_excluded_from_base_resolution(self):
        """pandas's hypothesis requirement is extra=='test'-gated; a base install
        (no extras requested) must not pull it into the graph."""
        deps = [ne.PinnedDependency("pandas", "2.2.1")]
        resolved, findings = ne.resolve_transitive_graph(deps, REQ_PY_311)
        assert "hypothesis" not in resolved


class TestExtrasInTransitiveGraph:
    """A pin like pandas[test] must pull the extra's own requirements into the graph."""

    def test_extra_requirements_are_walked(self):
        deps = [ne.PinnedDependency("pandas[test]", "2.2.1")]
        resolved, findings = ne.resolve_transitive_graph(deps, REQ_PY_311)
        assert findings == []
        assert resolved["hypothesis"] == "6.100.0"
        assert "sortedcontainers" in resolved  # reachable only through the extra

    def test_base_pin_still_excludes_extra_requirements(self):
        deps = [ne.PinnedDependency("pandas", "2.2.1")]
        resolved, _ = ne.resolve_transitive_graph(deps, REQ_PY_311)
        assert "hypothesis" not in resolved

    def test_extras_variant_is_not_reported_as_a_separate_package(self):
        deps = [ne.PinnedDependency("pandas[test]", "2.2.1")]
        resolved, _ = ne.resolve_transitive_graph(deps, REQ_PY_311)
        assert not [name for name in resolved if "[" in name]
        assert resolved["pandas"] == "2.2.1"

    def test_every_requested_extra_is_walked(self):
        deps = [ne.PinnedDependency("multi-extra-pkg[a,b]", "1.0.0")]
        resolved, findings = ne.resolve_transitive_graph(deps, REQ_PY_311)
        assert findings == []
        assert {"alpha-dep", "beta-dep", "core-dep"} <= set(resolved)

    def test_only_the_requested_extra_is_walked(self):
        deps = [ne.PinnedDependency("multi-extra-pkg[a]", "1.0.0")]
        resolved, _ = ne.resolve_transitive_graph(deps, REQ_PY_311)
        assert "alpha-dep" in resolved
        assert "beta-dep" not in resolved
        assert "core-dep" in resolved

    def test_extra_named_by_a_transitive_requirement_is_walked(self):
        deps = [ne.PinnedDependency("meta-pkg", "1.0.0")]
        resolved, findings = ne.resolve_transitive_graph(deps, REQ_PY_311)
        assert findings == []
        assert "hypothesis" in resolved  # meta-pkg -> pandas[test] -> hypothesis

    def test_conflict_created_by_an_extra_is_reported(self):
        deps = [
            ne.PinnedDependency("pandas[test]", "2.2.1"),
            ne.PinnedDependency("hypothesis", "5.0.0"),  # pandas[test] needs >=6.46.1
        ]
        resolved, findings = ne.resolve_transitive_graph(deps, REQ_PY_311)
        assert resolved is None
        assert findings and all(f.signal == "conflict" and f.severity == "confirmed" for f in findings)

    def test_unknown_extra_adds_nothing_and_does_not_fail(self):
        deps = [ne.PinnedDependency("pandas[nonexistent]", "2.2.1")]
        resolved, findings = ne.resolve_transitive_graph(deps, REQ_PY_311)
        assert findings == []
        assert "hypothesis" not in resolved
        assert resolved["pandas"] == "2.2.1"

    def test_signals_reach_packages_only_reachable_through_the_extra(self):
        with_extra = ne.check_transitive_signals(
            [ne.PinnedDependency("pandas[test]", "2.2.1")], REQ_PY_311)
        assert [f for f in with_extra if f.signal == "yanked" and f.package == "sortedcontainers"]

        without = ne.check_transitive_signals(
            [ne.PinnedDependency("pandas", "2.2.1")], REQ_PY_311)
        assert not [f for f in without if f.package == "sortedcontainers"]

    def test_all_requested_extras_are_parsed(self):
        name, extras = ne._split_pin_extras("multi-extra-pkg[b,a]")
        assert name == "multi-extra-pkg"
        assert extras == frozenset({"a", "b"})


class TestCheckTransitiveSignals:
    def test_direct_pins_are_skipped(self):
        deps = [ne.PinnedDependency("pandas", "2.2.1")]
        findings = ne.check_transitive_signals(deps, REQ_PY_311)
        assert all(f.package != "pandas" for f in findings)

    def test_transitive_package_checked_against_resolved_version(self):
        deps = [ne.PinnedDependency("pandas", "2.2.1")]
        findings = ne.check_transitive_signals(deps, {"major": 3, "minor": 8})
        # numpy resolves to 1.26.4 here (only version satisfying pandas's non-3.11/3.12 branch);
        # 1.26.4 declares requires-python >=3.9, which doesn't cover 3.8.
        unsupported = [f for f in findings if f.signal == "unsupported_python" and f.package == "numpy"]
        assert len(unsupported) == 1

    def test_unresolvable_graph_short_circuits_to_conflict_findings(self):
        deps = [
            ne.PinnedDependency("pandas", "2.2.1"),
            ne.PinnedDependency("numpy", "2.5.3"),
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

CLEAN_DEP = ne.PinnedDependency("core-dep", "1.0.0")  # fake package with no findings of any kind


def _sha256_of(payload):
    """Independent of the implementation: canonical JSON of everything except dependency_hash."""
    body = {k: v for k, v in payload.items() if k != "dependency_hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _old_style_manifest():
    """Shaped and hashed the way a tool version that predates the local_modules field wrote it."""
    manifest = {
        "python_version": {"major": 3, "minor": 11},
        "dependencies": [CLEAN_DEP.to_dict()],
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
        result = ne.generate_production_blueprint([CLEAN_DEP])
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
        result = ne.generate_production_blueprint([CLEAN_DEP])
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
            CLEAN_DEP,
            ne.PinnedDependency("torch", "2.3.1+cu121"),  # local version: skipped by direct checks
            ne.PinnedDependency("pandas[test]", "2.2.1"),
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
        deps = [ne.PinnedDependency("torch", "2.3.1+cu121")]
        result = ne.generate_production_blueprint(deps)
        at_generation = [f.to_dict() for f in result["drift_report"].heuristic]
        path = _write_literal(tmp_path, result["drift_report"].manifest.to_dict())
        _, report = _check(path, capsys)

        assert [f["signal"] for f in at_generation] == ["unverifiable_custom_index"]
        # Same finding; the check additionally tags it against the baseline generation recorded.
        assert [f["baseline_status"] for f in report["heuristic"]] == ["known"]
        assert [{k: v for k, v in f.items() if k != "baseline_status"} for f in report["heuristic"]] == at_generation


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
        ne.generate_production_blueprint([CLEAN_DEP])
        assert "checks" in order and "hash" in order
        assert order.index("checks") < order.index("hash")


# ---------------------------------------------------------------------------
# Generation-time baseline: recorded findings, and new vs known at check time
# ---------------------------------------------------------------------------

import copy


def _generate_file(tmp_path, deps, python_version=None):
    result = ne.generate_production_blueprint(deps)
    manifest = result["drift_report"].manifest
    return _write_literal(tmp_path, manifest.to_dict()), manifest


def _change_world(monkeypatch, mutate):
    """Swaps in a copy of the fake PyPI that mutate() has altered -- 'time has passed'."""
    world = copy.deepcopy(FAKE_PACKAGES)
    mutate(world)
    monkeypatch.setattr(ne, "_fetch_pypi_json", make_fake_fetch(world))
    ne.fetch_pypi_version_metadata.cache_clear()
    ne.fetch_pypi_package_metadata.cache_clear()


def _by_signal(report, bucket, signal):
    return [f for f in report[bucket] if f["signal"] == signal]


REQUESTS_YANKED = ne.PinnedDependency("requests", "2.32.0")  # yanked (confirmed) + stale (heuristic)
STALE_ONLY = ne.PinnedDependency("stale-package", "1.0.0")   # stale (heuristic) only
OLD_NUMPY = ne.PinnedDependency("numpy", "1.26.4")           # major_bump (heuristic) only at 3.11


class TestBaselineKeys:
    @pytest.mark.parametrize("finding,expected", [
        (ne.DriftFinding("requests", "2.32.0", "yanked", "confirmed", "m"), ("yanked", "requests", "2.32.0")),
        (ne.DriftFinding("old-package", "0.9.0", "removed", "confirmed", "m"), ("removed", "old-package", "0.9.0")),
        (ne.DriftFinding("numpy", "1.26.4", "unsupported_python", "confirmed", "m"), ("unsupported_python", "numpy", "1.26.4")),
        (ne.DriftFinding("torch", "2.3.1+cu121", "unverifiable_custom_index", "heuristic", "m"),
         ("unverifiable_custom_index", "torch", "2.3.1+cu121")),
        (ne.DriftFinding("stale-package", "1.0.0", "stale", "heuristic", "m", {"days_since_last_release": 900}),
         ("stale", "stale-package")),
        (ne.DriftFinding("numpy", "1.26.4", "major_bump", "heuristic", "m", latest_version="2.5.3"),
         ("major_bump", "numpy", "2")),
        (ne.DriftFinding("numpy", "<2", "conflict", "confirmed", "m", parent="pandas"),
         ("conflict", "numpy", "<2", "pandas")),
    ])
    def test_key_holds_the_facts_that_define_the_problem(self, finding, expected):
        assert ne.finding_baseline_key(finding) == expected

    def test_stale_key_ignores_the_changing_day_count(self):
        a = ne.DriftFinding("p", "1", "stale", "heuristic", "m", {"days_since_last_release": 800})
        b = ne.DriftFinding("p", "1", "stale", "heuristic", "m", {"days_since_last_release": 900})
        assert ne.finding_baseline_key(a) == ne.finding_baseline_key(b)

    def test_a_newer_latest_major_is_a_different_major_bump(self):
        a = ne.DriftFinding("numpy", "1.26.4", "major_bump", "heuristic", "m", latest_version="2.5.3")
        b = ne.DriftFinding("numpy", "1.26.4", "major_bump", "heuristic", "m", latest_version="3.0.0")
        assert ne.finding_baseline_key(a) != ne.finding_baseline_key(b)

    @pytest.mark.parametrize("signal,severity", [
        ("tampered", "confirmed"), ("local_module_missing", "confirmed"),
        ("local_module_unverifiable", "error"), ("check_error", "error"),
    ])
    def test_findings_with_no_generation_time_counterpart_have_no_key(self, signal, severity):
        assert ne.finding_baseline_key(ne.DriftFinding("x", "1", signal, severity, "m")) is None

    def test_conflict_findings_carry_their_parent(self):
        deps = [ne.PinnedDependency("pandas", "2.2.1"), ne.PinnedDependency("numpy", "2.5.3")]
        _, findings = ne.resolve_transitive_graph(deps, REQ_PY_311)
        assert findings and all(f.parent is not None for f in findings)  # "" for a requirement from a direct pin
        assert any(f.parent == "pandas" for f in findings)


class TestGenerationRecordsBaseline:
    def test_findings_at_generation_are_recorded(self):
        baseline = ne.generate_production_blueprint([REQUESTS_YANKED])["drift_report"].manifest.baseline
        assert ("yanked", "requests", "2.32.0") in baseline.findings
        assert ("stale", "requests") in baseline.findings
        assert baseline.errors == ()

    def test_clean_generation_records_an_empty_baseline_not_none(self):
        manifest = ne.generate_production_blueprint([CLEAN_DEP])["drift_report"].manifest
        assert manifest.baseline == ne.Baseline()

    def test_packages_that_could_not_be_checked_are_recorded(self):
        deps = [ne.PinnedDependency("flaky-package", "1.0.0")]
        baseline = ne.generate_production_blueprint(deps)["drift_report"].manifest.baseline
        assert baseline.errors == ("flaky-package",)

    def test_generation_report_findings_are_not_tagged(self):
        report = ne.generate_production_blueprint([REQUESTS_YANKED])["drift_report"]
        assert report.confirmed and all(f.baseline_status is None for f in report.confirmed + report.heuristic)

    def test_baseline_is_covered_by_the_hash(self, tmp_path, capsys):
        _, manifest = _generate_file(tmp_path, [REQUESTS_YANKED])
        forged = manifest.to_dict()
        forged["baseline"] = {"version": 1, "findings": [], "errors": []}  # silences the recorded yank
        _, report = _check(_write_literal(tmp_path, forged, "forged.py"), capsys)
        assert _by_signal(report, "confirmed", "tampered")


class TestCheckDriftClassifiesAgainstBaseline:
    def test_unchanged_world_reports_everything_as_known(self, tmp_path, capsys):
        path, _ = _generate_file(tmp_path, [REQUESTS_YANKED])
        exit_code, report = _check(path, capsys)
        assert [f["baseline_status"] for f in report["confirmed"]] == ["known"]
        assert [f["baseline_status"] for f in report["heuristic"]] == ["known"]
        assert exit_code == 1  # a known confirmed finding still fails the check

    def test_new_confirmed_finding_is_tagged_new(self, tmp_path, monkeypatch, capsys):
        path, _ = _generate_file(tmp_path, [CLEAN_DEP])

        def yank(world):
            world["core-dep"]["versions"]["1.0.0"]["yanked"] = True
        _change_world(monkeypatch, yank)

        exit_code, report = _check(path, capsys)
        assert [f["baseline_status"] for f in _by_signal(report, "confirmed", "yanked")] == ["new"]
        assert exit_code == 1

    def test_known_heuristic_alone_does_not_fail_the_check(self, tmp_path, capsys):
        path, _ = _generate_file(tmp_path, [STALE_ONLY])
        exit_code, report = _check(path, capsys)
        assert [f["baseline_status"] for f in report["heuristic"]] == ["known"]
        assert exit_code == 0

    def test_new_heuristic_fails_the_check(self, tmp_path, monkeypatch, capsys):
        path, _ = _generate_file(tmp_path, [CLEAN_DEP])

        def go_stale(world):
            world["core-dep"]["releases"]["1.0.0"]["upload_time"] = "2020-01-01T00:00:00.000000Z"
        _change_world(monkeypatch, go_stale)

        exit_code, report = _check(path, capsys)
        assert [f["baseline_status"] for f in _by_signal(report, "heuristic", "stale")] == ["new"]
        assert exit_code == 1

    def test_same_signal_with_different_facts_is_new(self, tmp_path, monkeypatch, capsys):
        path, _ = _generate_file(tmp_path, [OLD_NUMPY])  # major_bump: latest major was 2

        def newer_major(world):
            world["numpy"]["latest_version"] = "3.0.0"
            world["numpy"]["releases"]["3.0.0"] = {"upload_time": "2026-09-01T00:00:00.000000Z", "yanked": False}
        _change_world(monkeypatch, newer_major)

        _, report = _check(path, capsys)
        assert [f["baseline_status"] for f in _by_signal(report, "heuristic", "major_bump")] == ["new"]

    def test_finding_on_a_package_that_errored_at_generation_is_not_claimed_new(self, tmp_path, monkeypatch, capsys):
        path, _ = _generate_file(tmp_path, [ne.PinnedDependency("flaky-package", "1.0.0")])

        def recovers(world):
            world["flaky-package"] = copy.deepcopy(world["stale-package"])
        _change_world(monkeypatch, recovers)

        exit_code, report = _check(path, capsys)
        stale = _by_signal(report, "heuristic", "stale")
        assert [f["baseline_status"] for f in stale] == ["not_checked_at_generation"]
        assert exit_code == 1

    def test_new_findings_are_listed_before_known_ones(self, tmp_path, monkeypatch, capsys):
        path, _ = _generate_file(tmp_path, [REQUESTS_YANKED, CLEAN_DEP])

        def yank(world):
            world["core-dep"]["versions"]["1.0.0"]["yanked"] = True
        _change_world(monkeypatch, yank)

        _, report = _check(path, capsys)
        assert [(f["package"], f["baseline_status"]) for f in report["confirmed"]] == [
            ("core-dep", "new"), ("requests", "known"),
        ]

    def test_console_report_tags_each_finding(self, tmp_path, monkeypatch, capsys):
        path, _ = _generate_file(tmp_path, [REQUESTS_YANKED, CLEAN_DEP])

        def yank(world):
            world["core-dep"]["versions"]["1.0.0"]["yanked"] = True
        _change_world(monkeypatch, yank)

        ne.run_check_drift_pipeline(str(path))
        out = capsys.readouterr().out
        assert "[new]" in out and "[known]" in out

    def test_json_summarizes_the_split(self, tmp_path, monkeypatch, capsys):
        path, _ = _generate_file(tmp_path, [REQUESTS_YANKED, CLEAN_DEP])

        def yank(world):
            world["core-dep"]["versions"]["1.0.0"]["yanked"] = True
        _change_world(monkeypatch, yank)

        _, report = _check(path, capsys)
        assert report["baseline"] == {"recorded": True, "new": 1, "known": 2, "not_checked_at_generation": 0}

    def test_manifest_without_a_baseline_is_reported_flat_as_before(self, tmp_path, capsys):
        manifest = _old_style_manifest()
        manifest["dependencies"] = [STALE_ONLY.to_dict()]
        manifest["dependency_hash"] = _sha256_of(manifest)
        path = _write_literal(tmp_path, manifest)

        exit_code, report = _check(path, capsys)
        assert report["heuristic"] and all("baseline_status" not in f for f in report["heuristic"])
        assert report["baseline"] == {"recorded": False}
        assert exit_code == 1

        ne.run_check_drift_pipeline(str(path))
        assert "no generation-time baseline" in capsys.readouterr().out.lower()

    def test_unrecognized_baseline_format_is_treated_as_no_baseline(self, tmp_path, capsys):
        manifest = _old_style_manifest()
        manifest["dependencies"] = [STALE_ONLY.to_dict()]
        manifest["baseline"] = {"version": 99, "findings": [["stale", "stale-package"]], "errors": []}
        manifest["dependency_hash"] = _sha256_of(manifest)

        exit_code, report = _check(_write_literal(tmp_path, manifest), capsys)
        assert report["baseline"] == {"recorded": False}
        assert all("baseline_status" not in f for f in report["heuristic"])
        assert exit_code == 1


# ---------------------------------------------------------------------------
# Known custom sources: a not_found_on_pypi finding already present at generation is
# an expected state (private/custom-index package), not drift.
# ---------------------------------------------------------------------------

PRIVATE_PKG = ne.PinnedDependency("my-private-pkg", "1.0.0")  # not on the fake PyPI


class TestKnownCustomSources:
    def test_generation_still_reports_it_as_a_confirmed_finding(self):
        report = ne.generate_production_blueprint([PRIVATE_PKG])["drift_report"]
        assert [f.signal for f in report.confirmed] == ["not_found_on_pypi"]
        assert report.confirmed[0].baseline_status is None

    def test_known_one_is_a_notice_and_does_not_fail_the_check(self, tmp_path, capsys):
        path, _ = _generate_file(tmp_path, [PRIVATE_PKG])
        exit_code, report = _check(path, capsys)
        assert report["confirmed"] == []
        assert [(f["signal"], f["severity"], f["baseline_status"]) for f in report["notices"]] == [
            ("not_found_on_pypi", "notice", "known"),
        ]
        assert report["baseline"] == {"recorded": True, "new": 0, "known": 1, "not_checked_at_generation": 0}
        assert exit_code == 0

    def test_console_report_lists_it_as_a_notice_and_stays_clean(self, tmp_path, capsys):
        path, _ = _generate_file(tmp_path, [PRIVATE_PKG])
        exit_code = ne.run_check_drift_pipeline(str(path))
        out = capsys.readouterr().out
        assert "CUSTOM SOURCES" in out and "[known] [not_found_on_pypi]" in out
        assert "CONFIRMED ISSUES" not in out
        assert "STATUS: ✅ Clean" in out
        assert exit_code == 0

    def test_a_package_that_vanished_from_pypi_since_generation_still_fails(self, tmp_path, monkeypatch, capsys):
        path, _ = _generate_file(tmp_path, [CLEAN_DEP])

        def vanishes(world):
            del world["core-dep"]
        _change_world(monkeypatch, vanishes)

        exit_code, report = _check(path, capsys)
        assert [(f["signal"], f["baseline_status"]) for f in report["confirmed"]] == [("not_found_on_pypi", "new")]
        assert report["notices"] == []
        assert exit_code == 1

    def test_a_real_new_finding_still_fails_alongside_a_known_custom_source(self, tmp_path, monkeypatch, capsys):
        path, _ = _generate_file(tmp_path, [PRIVATE_PKG, CLEAN_DEP])

        def yank(world):
            world["core-dep"]["versions"]["1.0.0"]["yanked"] = True
        _change_world(monkeypatch, yank)

        exit_code, report = _check(path, capsys)
        assert [f["signal"] for f in report["confirmed"]] == ["yanked"]
        assert [f["signal"] for f in report["notices"]] == ["not_found_on_pypi"]
        assert exit_code == 1

    def test_without_a_baseline_it_stays_a_confirmed_finding(self, tmp_path, capsys):
        manifest = _old_style_manifest()
        manifest["dependencies"] = [PRIVATE_PKG.to_dict()]
        manifest["dependency_hash"] = _sha256_of(manifest)
        exit_code, report = _check(_write_literal(tmp_path, manifest), capsys)
        assert [f["signal"] for f in report["confirmed"]] == ["not_found_on_pypi"]
        assert report["notices"] == []
        assert exit_code == 1


# ---------------------------------------------------------------------------
# Stable finding keys in JSON, and the batch aggregate validation section
# ---------------------------------------------------------------------------

class TestFindingKeyInJson:
    @pytest.mark.parametrize("finding,expected", [
        (ne.DriftFinding("requests", "2.32.0", "yanked", "confirmed", "m"), ["yanked", "requests", "2.32.0"]),
        (ne.DriftFinding("stale-package", "1.0.0", "stale", "heuristic", "m", {"days_since_last_release": 9}),
         ["stale", "stale-package"]),
        (ne.DriftFinding("", "", "tampered", "confirmed", "m"), ["tampered", "", ""]),
        (ne.DriftFinding("cookbook", "", "local_module_missing", "confirmed", "m"), ["local_module_missing", "cookbook", ""]),
        (ne.DriftFinding("numpy", "1.26.4", "check_error", "error", "m"), ["check_error", "numpy", "1.26.4"]),
    ])
    def test_every_finding_serializes_a_key(self, finding, expected):
        assert finding.to_dict()["key"] == expected

    def test_key_ignores_the_volatile_parts_of_a_finding(self):
        a = ne.DriftFinding("p", "1", "stale", "heuristic", "no release in 800 days", {"days_since_last_release": 800})
        b = ne.DriftFinding("p", "1", "stale", "heuristic", "no release in 900 days", {"days_since_last_release": 900})
        assert a.to_dict()["key"] == b.to_dict()["key"] and a.to_dict()["message"] != b.to_dict()["message"]

    def test_check_drift_json_carries_keys(self, tmp_path, capsys):
        path, _ = _generate_file(tmp_path, [REQUESTS_YANKED])
        _, report = _check(path, capsys)
        assert [f["key"] for f in report["confirmed"]] == [["yanked", "requests", "2.32.0"]]
        assert [f["key"] for f in report["heuristic"]] == [["stale", "requests"]]


FROZEN = {"requests": "requests==2.32.0", "numpy": "numpy==1.26.4", "core-dep": "core-dep==1.0.0"}


def _make_batch(tmp_path):
    def nb(rel, source):
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({
            "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [source]}],
            "metadata": {"kernelspec": {"language": "python", "name": "python3", "display_name": "Python 3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5,
        }), encoding="utf-8")
    nb("a.ipynb", "import requests\n")
    nb("sub/b.ipynb", "import requests\n")
    nb("c.ipynb", "import numpy\n")
    nb("d.ipynb", "import core_dep\n")
    return tmp_path


def _run_batch(root, capsys, fmt="text", write=True, only=None):
    args = argparse.Namespace(
        suffix=None, in_place=False, universal=None, output=write, output_dir=None, timeout=300, format=fmt,
    )
    ne.run_batch_pipeline(str(root), args, dict(FROZEN), {}, None)
    return capsys.readouterr().out


class TestBatchAggregateValidation:
    def test_json_groups_findings_across_notebooks(self, tmp_path, capsys):
        out = _run_batch(_make_batch(tmp_path), capsys, fmt="json")
        v = json.loads(out)["validation"]
        assert v["notebooks_checked"] == 4
        assert v["totals"] == {"confirmed": 1, "heuristic": 2, "errors": 0}

        confirmed = [f for f in v["findings"] if f["severity"] == "confirmed"]
        assert [(f["key"], f["notebooks"]) for f in confirmed] == [
            (["yanked", "requests", "2.32.0"], ["a.ipynb", "sub/b.ipynb"]),
        ]
        heuristic = {tuple(f["key"]): f["notebooks"] for f in v["findings"] if f["severity"] == "heuristic"}
        assert heuristic == {
            ("stale", "requests"): ["a.ipynb", "sub/b.ipynb"],
            ("major_bump", "numpy", "2"): ["c.ipynb"],
        }

    def test_json_lists_every_notebook_with_its_counts(self, tmp_path, capsys):
        v = json.loads(_run_batch(_make_batch(tmp_path), capsys, fmt="json"))["validation"]
        assert {n["path"]: (n["confirmed"], n["heuristic"], n["errors"]) for n in v["notebooks"]} == {
            "a.ipynb": (1, 1, 0), "sub/b.ipynb": (1, 1, 0), "c.ipynb": (0, 1, 0), "d.ipynb": (0, 0, 0),
        }

    def test_validation_block_is_stable_and_machine_independent(self, tmp_path, capsys):
        root = _make_batch(tmp_path)
        first = json.loads(_run_batch(root, capsys, fmt="json"))["validation"]
        second = json.loads(_run_batch(root, capsys, fmt="json"))["validation"]
        assert first == second
        assert str(tmp_path) not in json.dumps(first)  # relative paths only

    def test_findings_are_ordered_confirmed_first(self, tmp_path, capsys):
        v = json.loads(_run_batch(_make_batch(tmp_path), capsys, fmt="json"))["validation"]
        severities = [f["severity"] for f in v["findings"]]
        assert severities == sorted(severities, key=["confirmed", "heuristic", "error"].index)

    def test_console_section_replaces_the_per_notebook_one_liners(self, tmp_path, capsys, monkeypatch):
        warnings = []
        monkeypatch.setattr(ne.logger, "warning", lambda msg, *a, **k: warnings.append(str(msg)))
        out = _run_batch(_make_batch(tmp_path), capsys)
        assert "BATCH DEPENDENCY VALIDATION" in out
        assert "affects 2 notebook(s): a.ipynb, sub/b.ipynb" in out
        assert out.index("CONFIRMED ISSUES") < out.index("WORTH REVIEWING")
        assert not [w for w in warnings if "--check-drift on this file" in w]

    def test_clean_batch_says_so_in_one_line(self, tmp_path, capsys):
        root = _make_batch(tmp_path)
        for rel in ("a.ipynb", "sub/b.ipynb", "c.ipynb"):
            (root / rel).unlink()
        out = _run_batch(root, capsys)
        assert "1 notebook(s) checked, no issues found" in out
        v = json.loads(_run_batch(root, capsys, fmt="json"))["validation"]
        assert v["totals"] == {"confirmed": 0, "heuristic": 0, "errors": 0}

    def test_analysis_only_batch_has_no_validation(self, tmp_path, capsys):
        root = _make_batch(tmp_path)
        assert json.loads(_run_batch(root, capsys, fmt="json", write=False))["validation"] is None
        assert "BATCH DEPENDENCY VALIDATION" not in _run_batch(root, capsys, write=False)


# ---------------------------------------------------------------------------
# Typed Baseline, explicit DriftFinding fields, typed per-notebook counts
# ---------------------------------------------------------------------------

class TestBaselineType:
    def test_to_dict_is_the_persisted_shape(self):
        baseline = ne.Baseline(findings=(("yanked", "requests", "2.32.0"),), errors=("flaky-package",))
        assert baseline.to_dict() == {
            "version": 1, "findings": [["yanked", "requests", "2.32.0"]], "errors": ["flaky-package"],
        }

    def test_round_trips(self):
        baseline = ne.Baseline(findings=(("stale", "p"), ("yanked", "q", "1")), errors=("x", ""))
        assert ne.Baseline.from_dict(baseline.to_dict()) == baseline

    @pytest.mark.parametrize("raw", [
        None, [], "baseline", {"version": 99, "findings": [], "errors": []},
        {"version": 1, "findings": "abc", "errors": []}, {"version": 1, "findings": [["ok", 3]], "errors": []},
        {"version": 1, "findings": [], "errors": [3]}, {"version": 1, "errors": []},
    ])
    def test_unusable_baselines_read_as_none(self, raw):
        assert ne.Baseline.from_dict(raw) is None

    def test_build_baseline_returns_a_sorted_typed_record(self):
        findings = [
            ne.DriftFinding("requests", "2.32.0", ne.Signal.YANKED, ne.Severity.CONFIRMED, "m"),
            ne.DriftFinding("a-pkg", "1", ne.Signal.STALE, ne.Severity.HEURISTIC, "m"),
            ne.DriftFinding("flaky", "1", ne.Signal.CHECK_ERROR, ne.Severity.ERROR, "m"),
        ]
        assert ne.build_baseline(findings) == ne.Baseline(
            findings=(("stale", "a-pkg"), ("yanked", "requests", "2.32.0")), errors=("flaky",),
        )

    def test_manifest_holds_a_typed_baseline_and_persists_it_as_a_dict(self):
        manifest = ne.generate_production_blueprint([REQUESTS_YANKED])["drift_report"].manifest
        assert isinstance(manifest.baseline, ne.Baseline)
        assert manifest.to_dict()["baseline"] == manifest.baseline.to_dict()
        assert ne.SteadyPyManifest.from_literal(manifest.to_dict()).baseline == manifest.baseline

    def test_finding_keys_are_tuples_but_serialize_as_lists(self):
        finding = ne.DriftFinding("requests", "2.32.0", ne.Signal.YANKED, ne.Severity.CONFIRMED, "m")
        assert ne.finding_baseline_key(finding) == ("yanked", "requests", "2.32.0")
        assert ne.finding_identity_key(finding) == ("yanked", "requests", "2.32.0")
        assert finding.to_dict()["key"] == ["yanked", "requests", "2.32.0"]


class TestDriftFindingExplicitFields:
    def test_major_bump_carries_latest_version_as_a_field_and_in_json_details(self):
        (finding,) = ne.check_major_bump("numpy", "1.26.4")
        assert finding.latest_version == "2.5.3"
        assert finding.to_dict()["details"]["latest_version"] == "2.5.3"

    def test_conflict_carries_its_parent_as_a_field_and_in_json_details(self):
        deps = [ne.PinnedDependency("pandas", "2.2.1"), ne.PinnedDependency("numpy", "2.5.3")]
        _, findings = ne.resolve_transitive_graph(deps, REQ_PY_311)
        via_pandas = [f for f in findings if f.parent == "pandas"]
        assert via_pandas and via_pandas[0].to_dict()["details"]["parent"] == "pandas"
        assert [f.parent for f in findings if f.parent is not None]  # "" means a requirement from a direct pin

    def test_keys_read_the_explicit_fields(self):
        bump = ne.DriftFinding("numpy", "1.26.4", ne.Signal.MAJOR_BUMP, ne.Severity.HEURISTIC, "m", latest_version="3.0.0")
        conflict = ne.DriftFinding("numpy", "<2", ne.Signal.CONFLICT, ne.Severity.CONFIRMED, "m", parent="pandas")
        assert ne.finding_baseline_key(bump) == ("major_bump", "numpy", "3")
        assert ne.finding_baseline_key(conflict) == ("conflict", "numpy", "<2", "pandas")

    def test_display_only_details_stay_in_details(self):
        (finding,) = ne.check_staleness("stale-package", "1.0.0")
        assert "days_since_last_release" in finding.details and finding.latest_version is None and finding.parent is None


class TestNotebookValidationCounts:
    def test_batch_validation_holds_typed_per_notebook_counts(self):
        report = ne.generate_production_blueprint([REQUESTS_YANKED])["drift_report"]
        validation = ne.build_batch_validation([("a.ipynb", report)])
        (counts,) = validation.notebooks
        assert isinstance(counts, ne.NotebookValidationCounts)
        assert (counts.path, counts.confirmed, counts.heuristic, counts.errors) == ("a.ipynb", 1, 1, 0)
        assert counts.to_dict() == {"path": "a.ipynb", "confirmed": 1, "heuristic": 1, "errors": 0}
        assert validation.to_dict()["notebooks"] == [counts.to_dict()]

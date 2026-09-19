"""
Live PyPI integration test for the drift-check PyPI client.

NOT part of the regular suite -- makes real network calls, so it's skipped
by default. Run explicitly with:

    RUN_LIVE_PYPI_TESTS=1 pytest test_drift_check_live.py -v

Purpose is narrow and deliberate: catch upstream PyPI JSON API schema drift
(a field renamed, a shape changed) that test_drift_check.py's mocked suite
cannot see, since it mocks the exact shape this test proves is still real.
This is NOT a place to re-test check-function logic -- that's the mocked
suite's job, and duplicating it here would just make CI flaky for no benefit.

Every assertion below is a durable, historical fact -- something true at
publication time that cannot change later (a specific release's declared
requires_python, a yank that already happened, a name that doesn't exist).
Nothing here asserts "latest version" or "is currently stale" or anything
else that legitimately changes over time; a live test asserting a moving
target breaks on schedule, not on regression.
"""

import os
import pytest

import notebook_env as ne

pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_LIVE_PYPI_TESTS"),
    reason="Live PyPI test skipped by default -- set RUN_LIVE_PYPI_TESTS=1 to run.",
)


@pytest.fixture(autouse=True)
def _clear_caches():
    ne.fetch_pypi_version_metadata.cache_clear()
    ne.fetch_pypi_package_metadata.cache_clear()
    yield


class TestLiveVersionMetadataShape:
    def test_known_yanked_release(self):
        """requests 2.32.0 was yanked for CVE-2024-35195 -- permanent historical record."""
        meta = ne.fetch_pypi_version_metadata("requests", "2.32.0")
        assert meta.status == "found"
        assert meta.yanked is True
        assert meta.yanked_reason and "35195" in meta.yanked_reason

    def test_known_requires_python(self):
        """A published release's own requires_python does not change after the fact."""
        meta = ne.fetch_pypi_version_metadata("pandas", "2.2.1")
        assert meta.status == "found"
        assert meta.requires_python == ">=3.9"

    def test_known_requires_dist_contains_expected_dependency(self):
        meta = ne.fetch_pypi_version_metadata("pandas", "2.2.1")
        assert meta.status == "found"
        assert any(r.startswith("numpy") for r in meta.requires_dist)

    def test_nonexistent_version_of_real_package(self):
        meta = ne.fetch_pypi_version_metadata("pandas", "999.999.999")
        assert meta.status == "not_found"

    def test_nonexistent_package(self):
        meta = ne.fetch_pypi_version_metadata("fake_pkg_does_not_exist_xyz123", "1.0.0")
        assert meta.status == "not_found"


class TestLivePackageMetadataShape:
    def test_found_package_has_releases_and_latest_version(self):
        meta = ne.fetch_pypi_package_metadata("pandas")
        assert meta.status == "found"
        assert meta.latest_version
        assert "2.2.1" in meta.releases
        assert meta.releases["2.2.1"]["upload_time"]

    def test_yanked_release_visible_in_package_level_releases(self):
        meta = ne.fetch_pypi_package_metadata("requests")
        assert meta.status == "found"
        assert meta.releases["2.32.0"]["yanked"] is True

    def test_nonexistent_package(self):
        meta = ne.fetch_pypi_package_metadata("fake_pkg_does_not_exist_xyz123")
        assert meta.status == "not_found"


class TestLiveRemovedVsYankedDisambiguation:
    """The exact real-world scenario that motivated the two-endpoint design:
    a 404 alone is ambiguous, and disambiguating requires both endpoints."""

    def test_version_removed_but_project_alive(self):
        version_meta = ne.fetch_pypi_version_metadata("pandas", "999.999.999")
        package_meta = ne.fetch_pypi_package_metadata("pandas")
        assert version_meta.status == "not_found"
        assert package_meta.status == "found"

    def test_whole_project_removed(self):
        version_meta = ne.fetch_pypi_version_metadata("fake_pkg_does_not_exist_xyz123", "1.0.0")
        package_meta = ne.fetch_pypi_package_metadata("fake_pkg_does_not_exist_xyz123")
        assert version_meta.status == "not_found"
        assert package_meta.status == "not_found"


class TestLiveTransitiveResolution:
    def test_real_conflict_detected(self):
        """pandas 2.2.1 declares numpy<2 in every marker branch; pinning numpy>=2.5
        alongside it is a genuine, permanent conflict."""
        deps = [
            {"name": "pandas", "version": "2.2.1", "flags": []},
            {"name": "numpy", "version": "2.5.0", "flags": []},
        ]
        resolved, findings = ne.resolve_transitive_graph(deps, {"major": 3, "minor": 11})
        assert resolved is None
        assert any(f.signal == "conflict" for f in findings)

    def test_real_resolvable_graph(self):
        deps = [{"name": "pandas", "version": "2.2.1", "flags": []}]
        resolved, findings = ne.resolve_transitive_graph(deps, {"major": 3, "minor": 11})
        assert findings == []
        assert resolved["pandas"] == "2.2.1"
        assert "numpy" in resolved
        assert "hypothesis" not in resolved  # test-extra must not leak into a base resolve

    def test_real_extra_is_expanded(self):
        """pandas 2.2.1 declares hypothesis under extra == "test" (historical, immutable)."""
        deps = [{"name": "pandas[test]", "version": "2.2.1", "flags": []}]
        resolved, findings = ne.resolve_transitive_graph(deps, {"major": 3, "minor": 11})
        assert findings == []
        assert resolved["pandas"] == "2.2.1"
        assert "hypothesis" in resolved
        assert not [name for name in resolved if "[" in name]
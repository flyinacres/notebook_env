"""
Tests for cell-magic and shell-install harvesting (harvest_cell_magics_and_commands,
harvest_index_urls_from_sources, classify_cell_source).

Kept separate from test_notebook_env_fixtures.py / kitchen_sink.ipynb on purpose:
that fixture and its tests are about AST-level import extraction. This module is
about a different concern, regex/line-level scanning of %pip/!pip/%conda/apt-get/
index-url lines, and uses its own fixture (magic_sink.ipynb) so a change to one
concern doesn't risk breaking assertions that belong to the other.

Several tests below encode CORRECT/INTENDED behavior for known, not-yet-fixed bugs
(see DEVELOPMENT.md) and are marked xfail with a reason rather than written to match
current buggy output. If one of these starts passing, pytest will report XPASS,
which is the signal to remove the xfail marker, the bug's been fixed.
"""

from pathlib import Path

import pytest

import notebook_env as ne


FIXTURE_DIR = Path(__file__).parent / "fixtures"
MAGIC_SINK_PATH = FIXTURE_DIR / "magic_sink.ipynb"


@pytest.fixture
def magic_sink_notebook():
    if not MAGIC_SINK_PATH.exists():
        pytest.fail(f"Fixture notebook not found at {MAGIC_SINK_PATH}.")
    return MAGIC_SINK_PATH


# =====================================================================
# UNIT TESTS: harvest_cell_magics_and_commands, isolated source strings
# =====================================================================

class TestPackageHarvesting:
    def test_plain_multi_package_pip_magic(self) -> None:
        pkgs, _, _, _, _ = ne.harvest_cell_magics_and_commands(
            ["%pip install gdown awscli"]
        )
        assert pkgs == {"gdown", "awscli"}

    def test_quoted_version_specifier_and_ignorable_flag(self) -> None:
        pkgs, _, _, _, _ = ne.harvest_cell_magics_and_commands(
            ['!pip install "spacy>=3.0" -q']
        )
        assert pkgs == {"spacy"}

    def test_requirements_file_reference_warns_not_silently_followed(self) -> None:
        pkgs, _, _, warnings, _ = ne.harvest_cell_magics_and_commands(
            ["%pip install -r requirements.txt"]
        )
        assert pkgs == set()
        assert any("requirements.txt" in w for w in warnings)

    def test_requirement_long_flag_also_warns(self) -> None:
        pkgs, _, _, warnings, _ = ne.harvest_cell_magics_and_commands(
            ["!pip install --requirement requirements.txt"]
        )
        assert pkgs == set()
        assert len(warnings) == 1

    def test_commented_out_pip_line_not_harvested(self) -> None:
        pkgs, _, _, _, _ = ne.harvest_cell_magics_and_commands(
            ["# !pip install should-not-be-harvested"]
        )
        assert pkgs == set()

    def test_bare_pip_inside_bash_cell_chained_with_apt_get(self) -> None:
        source = "%%bash\napt-get update && apt-get install -y graphviz\npip install kaggle-environments"
        pkgs, _, _, _, notices = ne.harvest_cell_magics_and_commands([source])
        assert pkgs == {"kaggle-environments"}
        assert any("apt-get" in n for n in notices)

    def test_conda_install_produces_notice_not_a_package(self) -> None:
        pkgs, _, _, _, notices = ne.harvest_cell_magics_and_commands(
            ["%conda install -c conda-forge lightgbm"]
        )
        # conda packages are intentionally NOT correlated against pip freeze,
        # so they must not show up in the pip-oriented harvested_packages set.
        assert "lightgbm" not in pkgs
        assert any("conda" in n.lower() for n in notices)

    def test_system_package_manager_call_outside_bash_cell(self) -> None:
        pkgs, _, _, _, notices = ne.harvest_cell_magics_and_commands(
            ["!yum install -y some-system-lib"]
        )
        assert pkgs == set()
        assert any("some-system-lib" in n for n in notices)

    @pytest.mark.xfail(
        reason="Known bug: '-e'/'--editable' is not in the value-consuming flag list, "
               "so the local path argument leaks into harvested_packages as a fake 'package'.",
        strict=True,
    )
    def test_editable_local_path_install_not_treated_as_a_package(self) -> None:
        pkgs, _, _, _, _ = ne.harvest_cell_magics_and_commands(
            ["!pip install -e ./local_package"]
        )
        assert "./local_package" not in pkgs

    @pytest.mark.xfail(
        reason="Known bug: same '-e' gap as above, applies to VCS install targets too.",
        strict=True,
    )
    def test_editable_vcs_install_not_treated_as_a_package(self) -> None:
        pkgs, _, _, _, _ = ne.harvest_cell_magics_and_commands(
            ["!pip install -e git+https://github.com/fake-org/fake-repo.git#egg=fakerepo"]
        )
        assert not any(p.startswith("git+") for p in pkgs)


class TestIndexUrlSeparation:
    def test_extra_index_url_only(self) -> None:
        _, base, extra, _, _ = ne.harvest_cell_magics_and_commands(
            ["!pip install torch --extra-index-url https://download.pytorch.org/whl/cu121"]
        )
        assert extra == {"https://download.pytorch.org/whl/cu121"}
        assert base == set()

    def test_base_index_url_only(self) -> None:
        _, base, extra, _, _ = ne.harvest_cell_magics_and_commands(
            ["!pip install torch --index-url https://custom.internal/simple"]
        )
        assert base == {"https://custom.internal/simple"}
        assert extra == set()

    def test_short_flag_base_index(self) -> None:
        _, base, extra, _, _ = ne.harvest_cell_magics_and_commands(
            ["!pip install foo -i https://custom.internal/simple"]
        )
        assert base == {"https://custom.internal/simple"}

    @pytest.mark.xfail(
        reason="Known bug: harvest_cell_magics_and_commands drops a --index-url match "
               "whenever --extra-index-url ALSO appears anywhere in the same line, "
               "even though they're two distinct flags with distinct URLs.",
        strict=True,
    )
    def test_base_and_extra_index_url_on_same_line_both_captured(self) -> None:
        _, base, extra, _, _ = ne.harvest_cell_magics_and_commands(
            ["!pip install onnxruntime --index-url https://custom.internal/simple "
             "--extra-index-url https://download.pytorch.org/whl/cu121"]
        )
        assert base == {"https://custom.internal/simple"}
        assert extra == {"https://download.pytorch.org/whl/cu121"}


class TestIndexUrlWrapperCompatibility:
    """
    harvest_index_urls_from_sources() is the older, single-set compatibility shim
    still used elsewhere in the pipeline (e.g. process_package_requirements). These
    tests document that it currently discards the base/extra distinction that
    harvest_cell_magics_and_commands() itself gets right.
    """

    def test_extra_index_url_passes_through(self) -> None:
        urls = ne.harvest_index_urls_from_sources(
            ["!pip install torch --extra-index-url https://download.pytorch.org/whl/cu121"]
        )
        assert urls == {"https://download.pytorch.org/whl/cu121"}

    @pytest.mark.xfail(
        reason="Known bug: when only --index-url (long form) is present and no "
               "--extra-index-url exists anywhere in the sources, the wrapper's fallback "
               "regex doesn't recognize the long form, so the URL is silently dropped "
               "entirely rather than surfaced (even mislabeled).",
        strict=True,
    )
    def test_base_index_url_alone_is_not_silently_dropped(self) -> None:
        urls = ne.harvest_index_urls_from_sources(
            ["!pip install foo --index-url https://custom.internal/simple"]
        )
        assert urls == {"https://custom.internal/simple"}

    @pytest.mark.xfail(
        reason="Known bug: when both --index-url and --extra-index-url are present, "
               "the wrapper returns only the extra one; the base override is dropped.",
        strict=True,
    )
    def test_both_present_wrapper_should_not_drop_base(self) -> None:
        urls = ne.harvest_index_urls_from_sources(
            ["!pip install onnxruntime --index-url https://custom.internal/simple "
             "--extra-index-url https://download.pytorch.org/whl/cu121"]
        )
        assert "https://custom.internal/simple" in urls
        assert "https://download.pytorch.org/whl/cu121" in urls


class TestCellClassification:
    def test_plain_python_cell(self) -> None:
        cell_type, clean = ne.classify_cell_source("import pandas as pd\n")
        assert cell_type == "PYTHON"
        assert "import pandas" in clean

    def test_bash_cell_header_stripped(self) -> None:
        cell_type, clean = ne.classify_cell_source("%%bash\napt-get install -y graphviz")
        assert cell_type == "SHELL_SCRIPT"
        assert "%%bash" not in clean
        assert "apt-get install" in clean

    def test_writefile_cell_header_stripped(self) -> None:
        cell_type, clean = ne.classify_cell_source("%%writefile helper.py\nimport requests")
        assert cell_type == "WRITEFILE"
        assert "%%writefile" not in clean
        assert "import requests" in clean

    def test_empty_source(self) -> None:
        cell_type, clean = ne.classify_cell_source("")
        assert cell_type == "PYTHON"
        assert clean == ""


# =====================================================================
# FIXTURE-BASED END-TO-END TEST: magic_sink.ipynb
# =====================================================================

class TestMagicSinkNotebook:
    def test_harvested_packages_from_full_notebook(self, magic_sink_notebook) -> None:
        success, imports, submodules, code_sources, err, lang, guarded, dyn_warns = (
            ne.extract_from_file(str(magic_sink_notebook))
        )
        assert success is True

        pkgs, base_urls, extra_urls, warnings, notices = ne.harvest_cell_magics_and_commands(
            code_sources
        )

        for expected in ("gdown", "awscli", "spacy", "kaggle-environments"):
            assert expected in pkgs, f"expected '{expected}' to be harvested"

        assert "should-not-be-harvested" not in pkgs
        assert "lightgbm" not in pkgs  # conda-only, must not be pip-correlated

        assert extra_urls == {"https://download.pytorch.org/whl/cu121"}
        assert any("requirements.txt" in w for w in warnings)
        assert any("conda" in n.lower() for n in notices)
        assert any("apt-get" in n or "yum" in n for n in notices)

    def test_writefile_gap_is_visible_via_ast_today(self, magic_sink_notebook) -> None:
        """
        Documents the related-but-separate gap noted in the fixture: today,
        extract_imports_from_sources() only strips lines starting with '%' or '!',
        so a %%writefile cell's body (meant to be written to disk, not executed)
        still gets AST-scanned and its imports folded in as if they were live,
        unconditional top-level imports. classify_cell_source() can already tell
        these cells apart, but that classification isn't used by the AST path yet.
        """
        success, imports, submodules, code_sources, err, lang, guarded, dyn_warns = (
            ne.extract_from_file(str(magic_sink_notebook))
        )
        assert success is True
        assert "requests" in imports
        assert "requests" not in guarded  # not guarded, just mislabeled as unconditional
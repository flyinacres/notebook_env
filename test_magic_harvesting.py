"""
Tests for cell-magic and shell-install harvesting (harvest_cell_magics_and_commands,
harvest_index_urls_from_sources, classify_cell_source).

Kept separate from test_notebook_env_fixtures.py / kitchen_sink.ipynb on purpose:
that fixture and its tests are about AST-level import extraction. This module is
about a different concern, regex/line-level scanning of %pip/!pip/%conda/apt-get/
index-url lines, and uses its own fixture (magic_sink.ipynb) so a change to one
concern doesn't risk breaking assertions that belong to the other.

As of notebook_env.py v30, all tests here assert correct/intended behavior directly.
The five cases that were previously marked xfail (editable/VCS install leaking into
harvested_packages; base vs. extra index URL conflation, both in the harvester itself
and in the harvest_index_urls_from_sources compatibility wrapper) are now fixed and
verified passing: PIP_VALUE_FLAGS + VCS_OR_PATH_PREFIXES close the editable/VCS leak,
and matching against the specific matched substring (rather than scanning the whole
line) fixed the base/extra conflation.
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

    def test_editable_local_path_install_not_treated_as_a_package(self) -> None:
        pkgs, _, _, _, _ = ne.harvest_cell_magics_and_commands(
            ["!pip install -e ./local_package"]
        )
        assert "./local_package" not in pkgs

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

    def test_base_and_extra_index_url_on_same_line_both_captured(self) -> None:
        _, base, extra, _, _ = ne.harvest_cell_magics_and_commands(
            ["!pip install onnxruntime --index-url https://custom.internal/simple "
             "--extra-index-url https://download.pytorch.org/whl/cu121"]
        )
        assert base == {"https://custom.internal/simple"}
        assert extra == {"https://download.pytorch.org/whl/cu121"}

# =====================================================================
# SCOPED FLAG ASSOCIATION & ORDER PRESERVATION
# =====================================================================

class TestScopedFlagAssociation:
    """Tests that pip flags are scoped strictly to specific command lines and document order is preserved."""

    def test_scoped_extra_index_url_association(self) -> None:
        """--extra-index-url attaches only to the package on that specific line."""
        sources = [
            "!pip install torch --extra-index-url https://download.pytorch.org/whl/cu121\n",
            "import pandas as pd\n",
            "import numpy as np\n"
        ]
        # v38 Contract: harvest returns packages paired with scoped flags
        pkg_flags_map = ne.harvest_scoped_cell_flags(sources)

        assert "torch" in pkg_flags_map
        assert "--extra-index-url" in pkg_flags_map["torch"]
        assert "https://download.pytorch.org/whl/cu121" in pkg_flags_map["torch"]

        # Standard imports must NOT inherit global index flags
        assert pkg_flags_map.get("pandas", []) == []
        assert pkg_flags_map.get("numpy", []) == []

    def test_first_encountered_order_preserved(self) -> None:
        """Packages maintain the chronological order they were encountered in code cells."""
        sources = [
            "import zstandard\n",
            "import astroid\n",
            "import pandas\n"
        ]
        imports, _, _, _ = ne.extract_imports_from_sources(sources)
        
        # v38 Contract: imports must preserve ['zstandard', 'astroid', 'pandas'] rather than sorting alphabetically
        assert list(imports) == ["zstandard", "astroid", "pandas"]

    def test_duplicate_flags_deduplicated_cleanly(self) -> None:
        """Repeated identical flags for the same package do not accumulate duplicates."""
        sources = [
            "!pip install torch --extra-index-url https://download.pytorch.org/whl/cu121\n",
            "!pip install torch --extra-index-url https://download.pytorch.org/whl/cu121\n"
        ]
        scoped = ne.harvest_scoped_cell_flags(sources)
        assert scoped["torch"] == ["--extra-index-url", "https://download.pytorch.org/whl/cu121"]

    def test_conflicting_base_index_url_last_wins(self) -> None:
        """Conflicting base --index-url flags resolve with last-encountered URL winning without accumulation."""
        sources = [
            "!pip install pkg --index-url https://first.index/simple\n",
            "!pip install pkg --index-url https://second.index/simple\n"
        ]
        scoped = ne.harvest_scoped_cell_flags(sources)
        assert scoped["pkg"] == ["--index-url", "https://second.index/simple"]

    def test_distinct_packages_maintain_independent_urls(self) -> None:
        """Different packages keep their own distinct index URLs."""
        sources = [
            "!pip install torch --extra-index-url https://download.pytorch.org/whl/cu121\n",
            "!pip install torchvision --extra-index-url https://vision.example.org/whl\n"
        ]
        scoped = ne.harvest_scoped_cell_flags(sources)
        assert scoped["torch"] == ["--extra-index-url", "https://download.pytorch.org/whl/cu121"]
        assert scoped["torchvision"] == ["--extra-index-url", "https://vision.example.org/whl"]

    def test_pip_occurrence_retains_version_and_flags(self) -> None:
        """Pip installs retain cell, line, raw token, version specifier, and scoped flags."""
        sources = [
            "!pip install torch==2.3.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121\n"
        ]
        occurrences = ne.harvest_pip_install_occurrences(sources)
        assert len(occurrences) == 1
        occ = occurrences[0]
        assert occ.name == "torch"
        assert occ.version_spec == "==2.3.1+cu121"
        assert occ.flags == ["--extra-index-url", "https://download.pytorch.org/whl/cu121"]
        assert occ.cell_idx == 0
        assert occ.line_idx == 0

    def test_atomic_last_wins_replaces_all_fields_indivisibly(self) -> None:
        """Later occurrence replaces version AND flags atomically without unioning earlier flags."""
        cell_0 = "!pip install foo==1.0 --index-url https://custom.repo/simple\n"
        cell_1 = "!pip install foo==2.0\n"  # No index url!
        
        occurrences = ne.harvest_pip_install_occurrences([cell_0, cell_1])
        resolved_map, conflict_warnings = ne.resolve_pip_occurrences(
            occurrences, is_execution_ordered=True
        )
        
        winning = resolved_map["foo"]
        assert winning.name == "foo"
        assert winning.version_spec == "==2.0"
        assert winning.flags == []  # Earlier --index-url is discarded!
        assert len(conflict_warnings) == 1
        assert "in execution sequence" in conflict_warnings[0]

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

    def test_base_index_url_alone_is_not_silently_dropped(self) -> None:
        urls = ne.harvest_index_urls_from_sources(
            ["!pip install foo --index-url https://custom.internal/simple"]
        )
        assert urls == {"https://custom.internal/simple"}

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

    def test_writefile_pip_commands_not_harvested(self) -> None:
        """Pip commands written inside %%writefile cells must not leak into harvested packages or scoped flags."""
        sources = [
            "%%writefile setup.py\n# Setup script\npip install dummy-pkg --extra-index-url https://writefile.example.com\n",
            "import pandas as pd\n"
        ]
        h_res = ne.harvest_cell_magics_and_commands(sources)
        assert "dummy-pkg" not in h_res.harvested_packages
        assert "https://writefile.example.com" not in h_res.extra_index_urls
        assert "dummy-pkg" not in h_res.scoped_flags

    def test_blank_line_padding_aligns_ast_lineno(self) -> None:
        """Replacing magics with blank lines ensures AST import line numbers match raw source lines."""
        cell_source = (
            "# Header comment (line 0)\n"
            "!pip install torch\n"          # line 1 (magic)
            "# Another comment (line 2)\n"
            "import torch\n"                 # line 3 (import)
        )
        import_occs = ne.extract_import_occurrences_from_source(cell_source, cell_idx=0)
        assert len(import_occs) == 1
        assert import_occs[0].module == "torch"
        assert import_occs[0].line_idx == 3


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


    def test_writefile_imports_isolated_from_primary_imports(self, magic_sink_notebook) -> None:
        """
        Verifies that imports inside %%writefile cells are isolated from primary notebook
        imports and extracted separately via extract_writefile_imports_from_sources.
        """
        success, imports, submodules, code_sources, err, lang, guarded, dyn_warns = (
            ne.extract_from_file(str(magic_sink_notebook))
        )
        assert success is True

        # Primary notebook imports should NOT include requests (which only lives in %%writefile)
        assert "requests" not in imports

        # requests SHOULD be captured via the writefile extraction helper
        writefile_imports = ne.extract_writefile_imports_from_sources(code_sources)
        assert "requests" in writefile_imports


class TestCellConsumingMagics:
    """Tests handling of cell-consuming magics (%%sql, %%html, %%R)."""

    def test_cell_consuming_magics_bypassed(self) -> None:
        """Cell-consuming magics like %%sql, %%html, and %%R bypass AST parsing cleanly."""
        sources = [
            "%%sql\nSELECT * FROM users WHERE age > 21;\n",
            "%%html\n<h1>Title</h1><p>Body text</p>\n",
            "%%R\nlibrary(ggplot2)\n"
        ]

        # Ensure classify_cell_source handles or AST parser skips them without SyntaxError
        for src in sources:
            cell_type, clean_body = ne.classify_cell_source(src)
            imports, submodules, guarded, warnings = ne.extract_imports_from_sources([src])
            assert imports == []
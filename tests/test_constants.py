"""Guards the signal/severity/status constants cleanup: production code must use the named constants,
not bare string literals, for these values (a typo in a literal silently never matches)."""
import re
from pathlib import Path

import notebook_env as ne

SOURCE = Path(ne.__file__).read_text(encoding="utf-8", errors="replace")
BARE = re.compile(r'\b(signal|severity|baseline_status|kind|status)\b(\s*(?:==|!=|=)\s*|\s+(?:not\s+)?in\s+\(?)"[a-z_]+"')


def _code_lines():
    """Lines of the module outside triple-quoted text (docstrings and the generated notebook cells)."""
    inside = False
    for number, line in enumerate(SOURCE.splitlines(), 1):
        quotes = line.count('"""')
        if not inside and quotes == 0:
            yield number, line
        if quotes % 2:
            inside = not inside


def test_no_bare_literals_for_signal_severity_or_status():
    offenders = [f"{n}: {line.strip()}" for n, line in _code_lines() if BARE.search(line)]
    assert offenders == []


def test_constant_values_are_the_wire_strings():
    assert ne.Signal.NOT_FOUND_ON_PYPI == "not_found_on_pypi"
    assert ne.Severity.NOTICE == "notice"
    assert ne.BaselineStatus.NOT_CHECKED_AT_GENERATION == "not_checked_at_generation"
    assert ne.DependencyStatus.DIRECT_REFERENCE == "direct_reference"
    assert ne.FetchStatus.NETWORK_ERROR == "network_error"
    assert ne.ReportKind.VALIDATION == "validation"


def test_constants_embed_in_the_manifest_literal_as_plain_strings():
    baseline = ne.build_baseline([ne.DriftFinding("requests", "2.32.0", ne.Signal.YANKED, ne.Severity.CONFIRMED, "m")])
    assert repr(baseline) == "{'version': 1, 'findings': [['yanked', 'requests', '2.32.0']], 'errors': []}"

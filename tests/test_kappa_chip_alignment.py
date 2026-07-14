"""
test_kappa_chip_alignment.py — guards the annotation UI against the value/label desync
that silently shifted Axis-A human labels by one category (kappa collapsed to ~0 while
the confusion matrix showed a clean off-diagonal).

Root cause it prevents: the annotation chips carry the label the human READS and the
`value` that gets STORED. If a reorder moves the labels but leaves the values (or the
JS category array) in a different order, every stored annotation is offset — a bug that
is invisible in the running app and only surfaces as a broken kappa. These assertions
fail the build if value != label, or if the chip set / JS arrays ever drift from the
canonical category enums.
"""
import re
from pathlib import Path

import rbac_benchmark
from rbac_benchmark.core.config import AWARENESS_CATEGORIES, LEVER_CATEGORIES

_KAPPA_HTML = Path(rbac_benchmark.__file__).parent / "server" / "templates" / "kappa.html"


def _chip_pairs(name: str) -> list[tuple[str, str]]:
    """Return (value, label) for every radio chip in the given axis group."""
    html = _KAPPA_HTML.read_text(encoding="utf-8")
    pat = re.compile(
        rf'name="{name}"\s+value="([A-Z_/]+)"[^>]*>\s*<label[^>]*>([A-Z_/ ]+)</label>'
    )
    return [(m.group(1), m.group(2).strip()) for m in pat.finditer(html)]


def _js_array(var: str) -> list[str]:
    html = _KAPPA_HTML.read_text(encoding="utf-8")
    m = re.search(rf"const {var}\s*=\s*\[([^\]]+)\]", html)
    assert m, f"{var} array not found in kappa.html"
    return re.findall(r"'([^']+)'", m.group(1))


def test_axis_a_chip_value_equals_label():
    pairs = _chip_pairs("axis-a")
    assert pairs, "no Axis-A chips found"
    for value, label in pairs:
        assert value == label, f"Axis-A chip value/label desync: value={value!r} label={label!r}"


def test_axis_a_chips_cover_awareness_categories():
    values = {v for v, _ in _chip_pairs("axis-a")}
    assert values == set(AWARENESS_CATEGORIES), (
        f"Axis-A chips {values} != AWARENESS_CATEGORIES {set(AWARENESS_CATEGORIES)}"
    )


def test_js_category_arrays_match_config_sets():
    # The JS arrays may be in a DIFFERENT display order than config, but they must cover
    # exactly the same label universe — a missing/renamed label mis-scores the matrix.
    assert set(_js_array("AWARENESS_CATS")) == set(AWARENESS_CATEGORIES)
    assert set(_js_array("LEVER_CATS")) == set(LEVER_CATEGORIES)

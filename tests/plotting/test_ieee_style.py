"""Tests for src.plotting.ieee_style — Phase 14 POL-01/02/03 Wave 0 helper.

Covers:
- Module exports (apply_ieee_style, OKABE_ITO, save_ieee_fig)
- Okabe-Ito palette structure (8 hex strings)
- rcParams mutations (savefig.dpi=300, image.cmap='cividis')
- PNG persistence via save_ieee_fig
- Tolerant PDF sibling export (does not raise)
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image

from src.plotting.ieee_style import OKABE_ITO, apply_ieee_style, save_ieee_fig


def test_imports_succeed():
    assert callable(apply_ieee_style)
    assert callable(save_ieee_fig)
    assert isinstance(OKABE_ITO, list)


def test_palette_is_eight_hex_strings():
    assert len(OKABE_ITO) == 8
    for c in OKABE_ITO:
        assert isinstance(c, str)
        assert c.startswith("#")
        assert len(c) == 7


def test_apply_sets_savefig_dpi_300():
    apply_ieee_style()
    assert matplotlib.rcParams["savefig.dpi"] == 300


def test_apply_sets_cividis_cmap():
    apply_ieee_style()
    assert matplotlib.rcParams["image.cmap"] == "cividis"


def test_save_ieee_fig_writes_png(tmp_path):
    apply_ieee_style()
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    target = tmp_path / "x.png"
    save_ieee_fig(fig, target)
    assert target.exists()
    assert target.stat().st_size > 0
    with Image.open(target) as im:
        assert im.size[0] > 0
        assert im.size[1] > 0
    plt.close(fig)


def test_save_ieee_fig_does_not_raise_on_pdf(tmp_path):
    apply_ieee_style()
    fig, ax = plt.subplots()
    ax.plot([1, 2], [3, 4])
    target = tmp_path / "y.png"
    # Must not raise even if PDF backend fails on some installs.
    save_ieee_fig(fig, target)
    plt.close(fig)

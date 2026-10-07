"""IEEE figure styling for the DS340 paper (Phase 14, POL-01/02/03).

Import this module at the top of any figure-generating script.

Usage:
    from src.plotting.ieee_style import apply_ieee_style, OKABE_ITO, save_ieee_fig
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.plot(x, y, label='LR')  # automatic colorblind colors + markers
    ax.set_xlabel('Training rows (count)')
    ax.set_ylabel('P&L ($)')
    ax.legend()
    save_ieee_fig(fig, 'experiments/figures/myfig.png')
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

try:
    import scienceplots  # noqa: F401  # registers 'science','ieee','no-latex'

    _HAS_SCIENCEPLOTS = True
except ImportError:
    _HAS_SCIENCEPLOTS = False

OKABE_ITO = [
    '#000000', '#E69F00', '#56B4E9', '#009E73',
    '#F0E442', '#0072B2', '#D55E00', '#CC79A7',
]


def apply_ieee_style() -> None:
    """Apply SciencePlots ['science','ieee','no-latex'] with Okabe-Ito color +
    variable linestyle + variable marker cycle. Sets savefig.dpi=300 and
    image.cmap='cividis'. Falls back to plain matplotlib with matching
    rcParams if scienceplots is not installed."""
    if _HAS_SCIENCEPLOTS:
        plt.style.use(['science', 'ieee', 'no-latex'])
    else:
        plt.style.use('default')
        mpl.rcParams.update({
            'font.family': 'serif',
            'font.size': 8,
            'axes.labelsize': 8,
            'axes.titlesize': 9,
            'xtick.labelsize': 7,
            'ytick.labelsize': 7,
            'legend.fontsize': 7,
            'figure.figsize': (3.5, 2.5),
            'figure.dpi': 300,
            'lines.linewidth': 1.0,
            'axes.grid': True,
            'grid.alpha': 0.3,
        })
    mpl.rcParams['axes.prop_cycle'] = (
        cycler(color=OKABE_ITO[:6])
        + cycler(linestyle=['-', '--', '-.', ':', '-', '--'])
        + cycler(marker=['o', 's', '^', 'D', 'v', 'x'])
    )
    mpl.rcParams['lines.markersize'] = 4
    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['image.cmap'] = 'cividis'


def save_ieee_fig(fig, path, dpi: int = 300) -> None:
    """Save figure at 300 DPI PNG to ``path``, also attempt a sibling PDF
    export (swallow PDF errors; PNG is canonical)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0.05)
    try:
        fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight', pad_inches=0.05)
    except Exception:
        pass

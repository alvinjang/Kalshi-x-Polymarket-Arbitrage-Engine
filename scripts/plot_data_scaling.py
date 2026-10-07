"""Plot the data-scaling experiment results.

Reads ``experiments/results/data_scaling/log.jsonl`` (produced by
``run_data_scaling.py``) and produces one PNG per metric:

  * rmse_vs_data.png
  * dir_acc_vs_data.png
  * pnl_at_2pp_vs_data.png

Each plot has one line per model tier so you can see visually whether
complexity (GRU/LSTM) earns its keep as more data becomes available.
This is the figure that answers the central CLAUDE.md research question.

Usage:
    python scripts/plot_data_scaling.py
    python scripts/plot_data_scaling.py --metric pnl_at_2pp --out out.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LOG_PATH = Path("experiments/results/data_scaling/log.jsonl")
OUT_DIR = Path("experiments/results/data_scaling")

_PLOT_METRICS = ("rmse", "dir_acc", "pnl_at_2pp", "pnl_at_3pp")


def _load_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _collect_series(rows: list[dict], metric: str) -> dict[str, list[tuple[int, float]]]:
    """Return {model_name: [(bars_per_pair, metric_value), ...]} sorted by bars."""
    series: dict[str, list[tuple[int, float]]] = {}
    for r in rows:
        bpp = int(r["bars_per_pair"])
        for name, m in r.get("metrics_by_model", {}).items():
            v = m.get(metric)
            if v is None:
                continue
            series.setdefault(name, []).append((bpp, float(v)))
    for name in series:
        series[name].sort(key=lambda x: x[0])
    return series


def plot_metric(rows: list[dict], metric: str, out_path: Path) -> None:
    """Plot one metric vs bars_per_pair for every model."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — pip install matplotlib to generate plots")
        return

    series = _collect_series(rows, metric)
    if not series:
        print(f"No data for metric '{metric}' — skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    for name, points in sorted(series.items()):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, marker="o", label=name)

    ax.set_xlabel("Training bars per pair")
    ax.set_ylabel(metric)
    ax.set_xscale("log")
    ax.set_title(f"Data scaling: {metric} vs training data size")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot data-scaling experiment results",
        prog="python scripts/plot_data_scaling.py",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=str(LOG_PATH),
        help=f"Path to scaling log JSONL (default: {LOG_PATH})",
    )
    parser.add_argument(
        "--metric",
        type=str,
        help="Plot only this metric (default: all of rmse, dir_acc, pnl_at_2pp, pnl_at_3pp)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(OUT_DIR),
        help=f"Output directory (default: {OUT_DIR})",
    )
    args = parser.parse_args(argv)

    rows = _load_log(Path(args.log))
    if not rows:
        print(f"No rows in {args.log} — run scripts/run_data_scaling.py first.")
        return 1

    out_dir = Path(args.out_dir)
    metrics = [args.metric] if args.metric else list(_PLOT_METRICS)
    for metric in metrics:
        out_path = out_dir / f"{metric}_vs_data.png"
        plot_metric(rows, metric, out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

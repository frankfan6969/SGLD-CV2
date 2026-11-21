"""Rebuild Gaussian figures from saved CSVs without rerunning sampling."""
import argparse
import csv
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cv2_matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = {"sgld": "SGLD", "cv1": "CV1", "cv2": "CV2", "ula": "Full-gradient ULA"}
COLORS = {"sgld": "#c44e52", "cv1": "#4c72b0", "cv2": "#55a868", "ula": "#8172b3"}


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def plot_results(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    config = manifest["config"]
    metrics = read_csv(folder / "metrics.csv")
    baselines = read_csv(folder / "reference_baselines.csv")
    controls = read_csv(folder / "analytic_controls.csv")
    def matching(rows, d, kappa):
        return [r for r in rows if int(r["d"]) == d and float(r["kappa"]) == kappa]
    for kappa in config["kappas"]:
        for axis in ["iteration", "total_seconds", "operations_observation_gradient_evaluations"]:
            dimensions = config["dimensions"]
            fig, axes = plt.subplots(1, len(dimensions), figsize=(5.3 * len(dimensions), 4.8), squeeze=False)
            for ax, d in zip(axes[0], dimensions):
                subset = matching(metrics, d, kappa)
                for method in config["methods"]:
                    selected = [r for r in subset if r["method"] == method]
                    xs, medians, lows, highs = [], [], [], []
                    for k in sorted({int(r["iteration"]) for r in selected}):
                        group = [r for r in selected if int(r["iteration"]) == k]
                        # A failed seed is not silently removed from the summary.
                        valid = all(r["status"] == "ok" for r in group)
                        values = [float(r["w2"]) for r in group] if valid else [np.nan]
                        xs.append(np.median([float(r.get(axis) or 0) for r in group]))
                        lows.append(np.quantile(values, 0.25))
                        medians.append(np.median(values))
                        highs.append(np.quantile(values, 0.75))
                    ax.plot(xs, medians, label=LABELS[method], color=COLORS[method],
                            linestyle="--" if method == "ula" else "-", linewidth=1.8)
                    ax.fill_between(xs, lows, highs, color=COLORS[method], alpha=0.12)
                baseline = [float(r["w2"]) for r in matching(baselines, d, kappa)]
                ax.axhspan(np.quantile(baseline, .25), np.quantile(baseline, .75), color="gray", alpha=.15)
                ax.axhline(np.median(baseline), color="gray", linestyle=":", label="IID vs IID baseline")
                if axis == "iteration":
                    control = matching(controls, d, kappa)
                    ax.plot([int(r["iteration"]) for r in control], [float(r["analytic_ula_w2"]) for r in control],
                            color="black", linestyle="-.", label="Analytic ULA / CV2 W2", linewidth=1.3)
                ax.set_title(f"d = {d}")
                ax.set_ylabel("W2 (original coordinates)")
                xlabel = {"iteration": "Iteration", "total_seconds": "Total ensemble seconds (includes setup)",
                          "operations_observation_gradient_evaluations": "Observation gradients (includes setup)"}[axis]
                ax.set_xlabel(xlabel)
                ax.grid(alpha=.2)
                ax.set_ylim(bottom=0)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", ncol=2 if len(dimensions) == 1 else 3, fontsize=8)
            fig.suptitle(f"Gaussian benchmark: kappa = {kappa:g}\nMedian and IQR across {len(config['seeds'])} paired seeds; R = {config['replicas']}", fontsize=11)
            fig.tight_layout(rect=(0, .18 if len(dimensions) == 1 else .13, 1, .9))
            fig.savefig(folder / f"w2_k{kappa:g}_{axis}.png", dpi=180)
            plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    plot_results(parser.parse_args().results)

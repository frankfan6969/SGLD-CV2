"""Paired replica bootstrap for changes between saved horizon checkpoints."""
import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from sgld_cv.randomness import generator
from sgld_cv.transport import make_accuracy_metric, load_accuracy_metric
from sgld_cv.workflow import METHODS, save_json


def diagnose(folder, output, *, pairs=((1024, 2048), (2048, 4096)), bootstraps=128,
             seed=2718, reference_size=None):
    folder = Path(folder)
    summary = json.loads((folder / "horizon_summary.json").read_text())
    spec = json.loads((folder / "run_config.json").read_text())["horizon_probe"]
    with np.load(folder / "reference_chains.npz") as data:
        longer = data["longer"].reshape(-1, data["longer"].shape[-1])
    d = longer.shape[1]
    if summary["metric"] == "w2":
        metric = make_accuracy_metric(d, kind="w2")
    else:
        metric = load_accuracy_metric(folder / "accuracy_calibration.npz", d=d)
    reference_size = spec["reference_size"] if reference_size is None else reference_size
    references = {s: longer[generator(s, 0, 84).choice(len(longer),
        reference_size, replace=False)] for s in summary["seed_list"]}
    rng = np.random.default_rng(seed)
    results = {}
    with threadpool_limits(limits=1):
        for method in METHODS:
            states = {}
            for s in summary["seed_list"]:
                with np.load(folder / f"probe_{method}_seed{s}.npz") as data:
                    states[s] = {int(k): x for k, x in zip(data["checkpoints"], data["states"])}
            results[method] = {}
            for start, end in pairs:
                if any(start not in states[s] or end not in states[s] for s in states):
                    raise ValueError(f"missing requested checkpoint {start} or {end}")
                differences = []
                boot = []
                for s, by_k in states.items():
                    ref = references[s]
                    differences.append(metric.distance(by_k[end], ref) -
                                       metric.distance(by_k[start], ref))
                for _ in range(bootstraps):
                    replicate = []
                    for s, by_k in states.items():
                        index = rng.integers(len(by_k[start]), size=len(by_k[start]))
                        ref = references[s]
                        replicate.append(metric.distance(by_k[end][index], ref) -
                                         metric.distance(by_k[start][index], ref))
                    boot.append(np.mean(replicate))
                results[method][f"{start}_to_{end}"] = {
                    "mean_accuracy_change": float(np.mean(differences)),
                    "conditional_bootstrap_q05": float(np.quantile(boot, .05)),
                    "conditional_bootstrap_q95": float(np.quantile(boot, .95)),
                    "seed_changes": differences}
    save_json(output, {"source_probe": str(folder), "metric": metric.kind,
        "reference_size": reference_size,
        "bootstraps": bootstraps, "bootstrap_seed": seed, "results": results,
        "interpretation": "Negative accuracy change is improvement. Paired resampling of independently seeded replicas is conditional on the two reference pools and the chosen seeds; intervals do not capture reference or between-seed uncertainty and cannot certify convergence."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstraps", type=int, default=128)
    parser.add_argument("--reference-size", type=int)
    args = parser.parse_args()
    diagnose(args.probe, args.output, bootstraps=args.bootstraps,
             reference_size=args.reference_size)


if __name__ == "__main__":
    main()

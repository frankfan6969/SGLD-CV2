"""Apply one predeclared mixing-and-moment rule to each 2D logistic method.

This is an exploratory step-size scan. Adaptive RMSprop ESS is reported but
cannot by itself certify a stationary posterior chain.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from diagnose_long_chains import diagnose
from sgld_cv.workflow import METHODS, save_json


SCALES = (1., .5, .25, .125, .0625, .03125)


def run(pilot, output, *, threshold=.1, seed=410300, reuse=None,
        reuse_scan=None, reference_diagnostics=None):
    pilot, output = Path(pilot), Path(output)
    if not 0 < threshold < 1:
        raise ValueError("threshold must lie between 0 and 1")
    output.mkdir(parents=True, exist_ok=False)
    explicit_reuse = {} if reuse is None else json.loads(Path(reuse).read_text())
    reuse = {}
    if reuse_scan is not None:
        previous = json.loads(Path(reuse_scan).read_text())
        for method, rows in previous["results"].items():
            reuse.setdefault(method, {}).update({str(row["scale"]): row["source"] for row in rows})
    for method, scales in explicit_reuse.items():
        reuse.setdefault(method, {}).update(scales)
    with np.load(pilot / "geometry.npz") as geometry:
        H = geometry["hessian"]
    settings = json.loads((pilot / "frozen_selection.json").read_text())["selection"]
    reference_path = (pilot / "reference_diagnostics.json" if reference_diagnostics is None
                      else Path(reference_diagnostics))
    reference = json.loads(reference_path.read_text())
    if not reference["reliable"]:
        raise ValueError("reference must pass diagnostics before step scanning")
    mean = np.asarray(reference["reference_mean"])
    covariance = np.asarray(reference["reference_covariance"])
    mean_guard = float(reference.get("standardized_mean_difference", 0.))
    covariance_guard = float(reference.get("standardized_covariance_difference", 0.))
    whiten = np.linalg.cholesky(H)
    selected = {}
    results = {}
    for method in METHODS:
        adaptive = method.startswith("p")
        base_draws = 90000 if adaptive else 30000
        results[method] = []
        for scale in SCALES:
            warmup, draws = round(5000 / scale), round(base_draws / scale)
            saved = reuse.get(method, {}).get(str(scale))
            if saved is not None:
                path = Path(saved)
                record = json.loads(path.read_text())
                if (record["warmup"] < warmup or record["draws_per_chain"] < draws or
                        record["seed"] != seed or record.get("step_scale", 1.) != scale or
                        method not in record["methods"]):
                    raise ValueError(f"reused chain has different settings: {path}")
            else:
                path = output / f"{method}_scale{scale:g}" / "summary.json"
                diagnose(pilot, path.parent, warmup=warmup, draws=draws,
                         seed=seed, methods=[method], step_scale=scale)
                record = json.loads(path.read_text())
            chain = record["methods"][method]
            diagnostics = chain["coordinate_diagnostics"]
            mean_error = float(np.linalg.norm((np.asarray(chain["mean"]) - mean) @ whiten) / np.sqrt(len(mean)))
            covariance_error = float(np.linalg.norm(whiten.T @
                (np.asarray(chain["covariance"]) - covariance) @ whiten) / len(mean))
            mixing_numbers_pass = bool(not chain["failures"] and
                diagnostics["max_rhat"] < 1.01 and
                min(diagnostics["min_bulk_ess"], diagnostics["min_tail_ess"]) >= 400)
            moment_pass = (mean_error + mean_guard <= threshold and
                           covariance_error + covariance_guard <= threshold)
            row = {"scale": scale, "step_size": settings[method]["step_size"] * scale,
                "warmup": record["warmup"], "draws_per_chain": record["draws_per_chain"],
                "max_rhat": diagnostics["max_rhat"],
                "min_bulk_ess": diagnostics["min_bulk_ess"],
                "min_tail_ess": diagnostics["min_tail_ess"],
                "mean_error_std": mean_error, "covariance_error_std": covariance_error,
                "mean_error_plus_reference_guard": mean_error + mean_guard,
                "covariance_error_plus_reference_guard": covariance_error + covariance_guard,
                "mixing_numbers_pass": mixing_numbers_pass,
                "moment_threshold_pass": moment_pass,
                "adaptive_ess_exploratory": adaptive,
                "source": str(path)}
            results[method].append(row)
            print(method, scale, "Rhat", round(row["max_rhat"], 4),
                  "ESS", round(row["min_bulk_ess"]),
                  "mean", round(mean_error, 4), "cov", round(covariance_error, 4), flush=True)
            save_json(output / "scan.json", {"source_pilot": str(pilot),
                "reference_diagnostics": str(reference_path),
                "reference_reliable": True, "moment_threshold_each": threshold,
                "reference_moment_guards": {"mean": mean_guard, "covariance": covariance_guard},
                "mixing_rule": "max Rhat < 1.01; min bulk and tail ESS >= 400",
                "adaptive_ess_policy": "numerical ESS reported, but exploratory for adaptive RMSprop",
                "candidate_scales": SCALES, "seed": seed,
                "results": results, "provisional_largest_passing_scale": selected,
                "interpretation": "Observed sample-moment errors have Monte Carlo and reference uncertainty. Selection remains provisional until size calibration and fresh-seed verification."})
            if mixing_numbers_pass and moment_pass and method not in selected:
                selected[method] = scale
            # If the largest candidate already passes, also measure one
            # smaller step for a local sensitivity check, while retaining
            # the largest passing step as the provisional choice.
            if method in selected and (selected[method] != 1. or scale < 1.):
                break
        if method not in selected:
            selected[method] = None
    save_json(output / "scan.json", {"source_pilot": str(pilot),
        "reference_diagnostics": str(reference_path),
        "reference_reliable": True, "moment_threshold_each": threshold,
        "reference_moment_guards": {"mean": mean_guard, "covariance": covariance_guard},
        "mixing_rule": "max Rhat < 1.01; min bulk and tail ESS >= 400",
        "adaptive_ess_policy": "numerical ESS reported, but exploratory for adaptive RMSprop",
        "candidate_scales": SCALES, "seed": seed,
        "results": results, "provisional_largest_passing_scale": selected,
        "interpretation": "Observed sample-moment errors have Monte Carlo and reference uncertainty. Selection remains provisional until size calibration and fresh-seed verification."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path,
        default=Path("results/stage456_validation/synthetic_logistic"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=.1)
    parser.add_argument("--reuse", type=Path)
    parser.add_argument("--reuse-scan", type=Path)
    parser.add_argument("--reference-diagnostics", type=Path)
    args = parser.parse_args()
    run(args.pilot, args.output, threshold=args.threshold, reuse=args.reuse,
        reuse_scan=args.reuse_scan, reference_diagnostics=args.reference_diagnostics)


if __name__ == "__main__":
    main()

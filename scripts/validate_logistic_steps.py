"""Fresh-seed check of the provisional six-method 2D logistic step sizes."""
import argparse
import json
from pathlib import Path

import numpy as np

from diagnose_long_chains import diagnose
from sgld_cv.workflow import METHODS, save_json


def run(scan_path, output, *, seed=410301):
    scan_path, output = Path(scan_path), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    scan = json.loads(scan_path.read_text())
    pilot = Path(scan["source_pilot"])
    reference = json.loads(Path(scan["reference_diagnostics"]).read_text())
    if not reference["reliable"]:
        raise ValueError("held-out check requires a validated reference")
    mean = np.asarray(reference["reference_mean"])
    covariance = np.asarray(reference["reference_covariance"])
    with np.load(pilot / "geometry.npz") as data:
        whiten = np.linalg.cholesky(data["hessian"])
    mean_guard = scan["reference_moment_guards"]["mean"]
    covariance_guard = scan["reference_moment_guards"]["covariance"]
    threshold = scan["moment_threshold_each"]
    results = {}
    for method in METHODS:
        scale = scan["provisional_largest_passing_scale"].get(method)
        if scale is None:
            raise ValueError(f"no provisional step for {method}")
        selected = next(r for r in scan["results"][method] if r["scale"] == scale)
        source = output / method
        diagnose(pilot, source, warmup=selected["warmup"],
            draws=selected["draws_per_chain"], seed=seed,
            methods=[method], step_scale=scale)
        result = json.loads((source / "summary.json").read_text())["methods"][method]
        diagnostic = result["coordinate_diagnostics"]
        mean_error = float(np.linalg.norm((np.asarray(result["mean"]) - mean) @ whiten) / np.sqrt(len(mean)))
        covariance_error = float(np.linalg.norm(whiten.T @
            (np.asarray(result["covariance"]) - covariance) @ whiten) / len(mean))
        mixing_numbers_pass = bool(not result["failures"] and
            diagnostic["max_rhat"] < 1.01 and
            min(diagnostic["min_bulk_ess"], diagnostic["min_tail_ess"]) >= 400)
        moment_pass = (mean_error + mean_guard <= threshold and
                       covariance_error + covariance_guard <= threshold)
        results[method] = {"scale": scale, "step_size": selected["step_size"],
            "warmup": selected["warmup"], "draws_per_chain": selected["draws_per_chain"],
            "max_rhat": diagnostic["max_rhat"],
            "min_bulk_ess": diagnostic["min_bulk_ess"],
            "min_tail_ess": diagnostic["min_tail_ess"],
            "mean_error_plus_reference_guard": mean_error + mean_guard,
            "covariance_error_plus_reference_guard": covariance_error + covariance_guard,
            "mixing_numbers_pass": mixing_numbers_pass,
            "moment_threshold_pass": moment_pass,
            "adaptive_ess_exploratory": method.startswith("p"),
            "source": str(source / "summary.json")}
        save_json(output / "validation.json", {"source_scan": str(scan_path),
            "fresh_seed": seed, "threshold_each": threshold,
            "results": results,
            "interpretation": "Fresh-seed moment/mixing check. Adaptive ESS remains exploratory. Passing this check does not determine final K or accuracy resolution."})
        print(method, "mixing", mixing_numbers_pass, "moments", moment_pass,
              "guarded covariance", round(covariance_error + covariance_guard, 4), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path,
        default=Path("results/logistic_step_scan_guarded/scan.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=410301)
    args = parser.parse_args()
    run(args.scan, args.output, seed=args.seed)


if __name__ == "__main__":
    main()

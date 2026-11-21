"""Combine fresh-seed logistic checks with longer pCV diagnostic chains."""
import argparse
import json
from pathlib import Path

import numpy as np

from sgld_cv.workflow import save_json


def run(validation, extended, reference, pilot, output):
    validation, extended, reference, pilot, output = map(
        Path, (validation, extended, reference, pilot, output))
    record = json.loads(validation.read_text())
    extra = json.loads(extended.read_text())
    ref = json.loads(reference.read_text())
    with np.load(pilot / "geometry.npz") as data:
        whiten = np.linalg.cholesky(data["hessian"])
    mean, covariance = np.asarray(ref["reference_mean"]), np.asarray(ref["reference_covariance"])
    results = dict(record["results"])
    threshold = record["threshold_each"]
    for method in ("pcv1", "pcv2"):
        chain = extra["methods"][method]
        diag = chain["coordinate_diagnostics"]
        mean_error = float(np.linalg.norm((np.asarray(chain["mean"]) - mean) @ whiten) /
                           np.sqrt(len(mean)) + ref["standardized_mean_difference"])
        covariance_error = float(np.linalg.norm(whiten.T @
            (np.asarray(chain["covariance"]) - covariance) @ whiten) /
            len(mean) + ref["standardized_covariance_difference"])
        results[method] = {**results[method],
            "original_fresh_seed_max_rhat": results[method]["max_rhat"],
            "max_rhat": diag["max_rhat"],
            "min_bulk_ess": diag["min_bulk_ess"],
            "min_tail_ess": diag["min_tail_ess"],
            "mean_error_plus_reference_guard": mean_error,
            "covariance_error_plus_reference_guard": covariance_error,
            "warmup": extra["warmup"], "draws_per_chain": extra["draws_per_chain"],
            "mixing_numbers_pass": bool(not chain["failures"] and diag["max_rhat"] < 1.01 and
                                        min(diag["min_bulk_ess"], diag["min_tail_ess"]) >= 400),
            "moment_threshold_pass": mean_error <= threshold and covariance_error <= threshold,
            "source": str(extended) + f"#methods.{method}"}
    save_json(output, {"source_validation": str(validation),
        "source_extended_validation": str(extended),
        "reference_diagnostics": str(reference),
        "threshold_each": threshold,
        "results": results,
        "all_numerical_checks_pass": all(r["mixing_numbers_pass"] and
            r["moment_threshold_pass"] for r in results.values()),
        "adaptive_ess_policy": "Numerical ESS for RMSprop methods remains exploratory",
        "interpretation": "Step-size diagnostic only. It does not establish a stationary posterior for adaptive methods or a final independent-replica horizon."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path,
        default=Path("results/logistic_step_validation/validation.json"))
    parser.add_argument("--extended", type=Path,
        default=Path("results/logistic_pcv_extended_validation/summary.json"))
    parser.add_argument("--reference", type=Path,
        default=Path("results/logistic_reference_refined/reference_diagnostics.json"))
    parser.add_argument("--pilot", type=Path,
        default=Path("results/stage456_validation/synthetic_logistic"))
    parser.add_argument("--output", type=Path,
        default=Path("results/logistic_step_validation/validated_steps.json"))
    args = parser.parse_args()
    run(args.validation, args.extended, args.reference, args.pilot, args.output)


if __name__ == "__main__":
    main()

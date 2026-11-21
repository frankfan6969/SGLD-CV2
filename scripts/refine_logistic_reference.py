"""Build longer independent HMC references for 2D logistic moment gates."""
import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from sgld_cv.anchors import supplied_anchor
from sgld_cv.data import synthetic_logistic
from sgld_cv.references import hmc_reference, validate_reference_pair
from sgld_cv.workflow import save_json


def run(pilot, output, *, warmup=1000, draws=5000, seed=7400):
    pilot, output = Path(pilot), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((pilot / "manifest.json").read_text())["config"]
    model, _ = synthetic_logistic(**{k: v for k, v in config["problem"].items() if k != "kind"})
    with np.load(pilot / "geometry.npz") as data:
        theta, H = data["anchor"], data["hessian"]
    anchor = supplied_anchor(theta, provenance="saved pilot mode")
    original = config["reference"]
    options = {"chains": original["chains"], "warmup": warmup,
        "step_size": original["step_size"],
        "leapfrog_steps": original["leapfrog_steps"]}
    with threadpool_limits(limits=1):
        first, first_diag = hmc_reference(model, anchor, draws=draws, seed=seed, **options)
        longer, longer_diag = hmc_reference(model, anchor, draws=2 * draws, seed=seed + 1, **options)
    agreement = validate_reference_pair(first, longer, first_diag, longer_diag)
    a = first.reshape(-1, model.d)
    b = longer.reshape(-1, model.d)
    whiten = np.linalg.cholesky(H)
    diagnostics = {"kind": "approximate full-data HMC",
        "first": first_diag, "longer": longer_diag, "agreement": agreement,
        "reliable": agreement["reliable"],
        "reference_mean": b.mean(axis=0),
        "reference_covariance": np.cov(b, rowvar=False),
        "calibration_mean": a.mean(axis=0),
        "calibration_covariance": np.cov(a, rowvar=False),
        "standardized_mean_difference": float(np.linalg.norm((a.mean(axis=0) - b.mean(axis=0)) @ whiten) / np.sqrt(model.d)),
        "standardized_covariance_difference": float(np.linalg.norm(whiten.T @
            (np.cov(a, rowvar=False) - np.cov(b, rowvar=False)) @ whiten) / model.d),
        "scoring_note": "This new reference is for diagnostic moments only; saved low-dimensional W2 probe references remain independently sampled from their original pool."}
    np.savez_compressed(output / "reference_chains.npz", first=first, longer=longer)
    save_json(output / "reference_diagnostics.json", diagnostics)
    print(json.dumps({k: diagnostics[k] for k in ("reliable", "standardized_mean_difference",
                                                   "standardized_covariance_difference")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path,
        default=Path("results/stage456_validation/synthetic_logistic"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=5000)
    args = parser.parse_args()
    run(args.pilot, args.output, draws=args.draws)


if __name__ == "__main__":
    main()

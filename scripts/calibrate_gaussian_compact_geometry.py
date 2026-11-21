"""Lean full-metric CV1/CV2 step calibration for four Gaussian factorial cells."""
import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from sgld_cv.anchors import supplied_anchor
from sgld_cv.costs import CostLedger
from sgld_cv.diagnostics import chain_diagnostics
from sgld_cv.estimators import prepare_estimator
from sgld_cv.gaussian import make_gaussian_target
from sgld_cv.randomness import generator
from sgld_cv.samplers import run_chains
from sgld_cv.transport import make_accuracy_metric
from sgld_cv.workflow import save_json


CONFIG = Path("configs/gaussian_compact_geometry_design.json")
OUTPUT = Path("results/gaussian_compact_geometry_calibration")


def full_metric(target, estimator):
    model = target.model
    cost = CostLedger()
    if estimator.method == "cv2":
        H = estimator.hessian_hat + model.prior_precision * np.eye(model.d)
    else:
        with cost.measure("full_metric_hessian"):
            H = model.full_hessian(np.zeros(model.d))
            cost.count("observation_hessian_evaluations", model.N)
            cost.count("dense_hessian_assemblies")
            cost.count("data_row_visits", model.N)
    np.testing.assert_allclose(H, target.precision, rtol=1e-9, atol=1e-7)
    with cost.measure("full_metric_inverse"):
        P = np.linalg.solve(H, np.eye(model.d))
        P = (P + P.T) / 2
    return P, cost


def main(*, batch_size=None, dimensions=None, output=None):
    config = json.loads(CONFIG.read_text())
    folder = OUTPUT if output is None else Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    if config["purpose"] != "gaussian_compact_geometry_calibration_then_frozen_confirmation":
        raise ValueError("wrong Gaussian design purpose")
    with threadpool_limits(limits=1):
        for d in (config["dimensions"] if dimensions is None else dimensions):
            for kappa in config["target_condition_numbers"]:
                target = make_gaussian_target(N=config["N"], d=d, kappa=kappa,
                    seed=config["calibration_design_seed"],
                    residual_scale=config["residual_scale"])
                metric = make_accuracy_metric(d, kind="standardized_sliced_w2",
                    center=target.mean, covariance=target.covariance,
                    projections=config["score_projections"],
                    seed=config["score_projection_seed"])
                initial = target.sample(config["confirmation_chains"],
                    generator(config["calibration_sampler_seed"], 0, 83)) * 2
                anchor = supplied_anchor(target.mean, provenance="known Gaussian target mean")
                for method in config["primary_methods"]:
                    estimator = prepare_estimator(target.model, method, anchor)
                    P, P_cost = full_metric(target, estimator)
                    for h in config["calibration_step_candidates"]:
                        path = folder / f"d{d}_k{kappa}_{method}_h{h:g}.json"
                        if path.exists():
                            continue
                        samples, run = run_chains(estimator, initial=initial,
                            step_size=h, batch_size=(config["primary_batch_size"] if batch_size is None else batch_size),
                            warmup=config["calibration_warmup"],
                            draws=config["calibration_draws_per_chain"],
                            chains=config["confirmation_chains"],
                            seed=config["calibration_sampler_seed"],
                            preconditioner="frozen_full", frozen_matrix=P,
                            metric_cost=P_cost)
                        if run.failures:
                            row = {"status": "nonfinite", "failures": run.failures}
                        else:
                            diag = chain_diagnostics(samples)
                            flat = samples.reshape(-1, d)
                            mean = np.linalg.norm((flat.mean(axis=0) - target.mean) @ metric.transform) / np.sqrt(d)
                            covariance = np.cov(flat, rowvar=False, ddof=0)
                            cov = np.linalg.norm(metric.transform.T @
                                (covariance - target.covariance) @ metric.transform) / d
                            row = {"status": "ok", "max_rhat": diag["max_rhat"],
                                "min_bulk_ess": diag["min_bulk_ess"],
                                "min_tail_ess": diag["min_tail_ess"],
                                "standardized_mean_error": float(mean),
                                "standardized_covariance_error": float(cov),
                                "cost": run.costs[-1]}
                        row.update({"d": d, "kappa": kappa, "design_seed": config["calibration_design_seed"],
                                    "sampler_seed": config["calibration_sampler_seed"],
                                    "method": method, "h": h,
                                    "batch_size": config["primary_batch_size"] if batch_size is None else batch_size,
                                    "target_metadata": target.metadata})
                        save_json(path, row)
                        print(d, kappa, method, h, row["status"],
                              row.get("max_rhat"), row.get("standardized_covariance_error"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--dimensions", type=int, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    main(batch_size=args.batch_size, dimensions=args.dimensions, output=args.output)

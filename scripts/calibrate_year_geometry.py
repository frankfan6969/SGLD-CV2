"""Small, reproducible Year ridge geometry and CV-order calibration."""
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from sgld_cv.anchors import supplied_anchor
from sgld_cv.costs import CostLedger
from sgld_cv.data import year_model
from sgld_cv.diagnostics import chain_diagnostics
from sgld_cv.estimators import prepare_estimator
from sgld_cv.randomness import generator
from sgld_cv.references import ridge_posterior
from sgld_cv.samplers import run_chains
from sgld_cv.transport import load_accuracy_metric, metric_fingerprint
from sgld_cv.workflow import save_json


PILOT = Path("results/stage456_validation/year_ridge")
CALIBRATION = Path("results/year_resolution_diagnostic/accuracy_calibration.npz")
OUTPUT = Path("results/year_targeted_calibration_v2")
SEEDS = (910, 911)
ARMS = (("cv1_h01", "cv1", .1), ("cv1_h01_long", "cv1", .1),
        ("cv1_h015", "cv1", .15),
        ("cv1_h02", "cv1", .2), ("cv1_h03", "cv1", .3),
        ("cv2_h03", "cv2", .3), ("cv2_h06", "cv2", .6))
WARMUP, DRAWS, CHAINS, BATCH = 1000, 4000, 4, 128
DRAWS_BY_ARM = {"cv1_h01_long": 16000, "cv1_h015": 12000, "cv1_h02": 8000}


def moment_errors(samples, posterior, metric):
    d = len(posterior.mean)
    mean = np.linalg.norm((samples.mean(axis=0) - posterior.mean) @ metric.transform) / np.sqrt(d)
    covariance = np.cov(samples, rowvar=False, ddof=0)
    cov = np.linalg.norm(metric.transform.T @
                         (covariance - posterior.covariance) @ metric.transform) / d
    return float(mean), float(cov)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((PILOT / "manifest.json").read_text())["config"]["problem"]
    model, data_meta = year_model(**{k: v for k, v in config.items() if k != "kind"})
    metric = load_accuracy_metric(CALIBRATION, d=model.d)
    with threadpool_limits(limits=1):
        posterior = ridge_posterior(model)
        with np.load(PILOT / "geometry.npz") as saved:
            anchor_delta = float(np.linalg.norm((posterior.mean - saved["anchor"]) @ metric.transform) /
                                 np.sqrt(model.d))
            precision_relative_difference = float(np.linalg.norm(posterior.precision - saved["hessian"]) /
                                                  np.linalg.norm(posterior.precision))
        anchor = supplied_anchor(posterior.mean, provenance="analytic ridge sufficient statistics",
                                 cost=posterior.cost)
        cv2 = prepare_estimator(model, "cv2", anchor)
        rng = generator(901, 0, 99)
        residuals = []
        for _ in range(8):
            theta = posterior.sample(1, rng)[0]
            batch = rng.choice(model.N, BATCH, replace=False)
            g = cv2.gradient(theta, batch)
            exact = posterior.precision @ (theta - posterior.mean)
            residuals.append(float(np.linalg.norm(g - exact) / max(np.linalg.norm(exact), 1)))
        eigenvalues = np.linalg.eigvalsh(posterior.precision)
        identity_h = json.loads((PILOT / "frozen_selection.json").read_text())["selection"]["cv2"]["step_size"]
        identity_efold = -1 / np.log1p(-identity_h * eigenvalues[0] / 2)
        metric_cost = CostLedger()
        with metric_cost.measure("full_metric_inverse"):
            inverse = posterior.covariance
        metric_cost.count("linear_solves")
        with metric_cost.measure("full_metric_cholesky_for_initialization"):
            L = np.linalg.cholesky(inverse)
        # For Gaussian CV2, full-H ULA has stationary covariance Sigma/(1-h/4).
        # This is a population discretization calculation, independent of Monte Carlo noise.
        analytic_bias = {str(h): float(h / (4 - h)) for h in (.1, .3, .6)}
        metadata = {"dataset": data_meta["dataset"], "rows": model.N, "dimension": model.d,
                    "scoring_fingerprint": metric_fingerprint(metric),
                    "anchor_standardized_difference_from_pilot": anchor_delta,
                    "precision_relative_difference_from_pilot": precision_relative_difference,
                    "max_relative_cv2_gradient_error": max(residuals),
                    "precision_min_eigenvalue": float(eigenvalues[0]),
                    "precision_max_eigenvalue": float(eigenvalues[-1]),
                    "precision_condition_number": float(eigenvalues[-1] / eigenvalues[0]),
                    "identity_cv2_pilot_h": identity_h,
                    "identity_cv2_slowest_efold_steps": float(identity_efold),
                    "full_cv2_stationary_variance_inflation": analytic_bias,
                    "full_metric_setup": metric_cost.snapshot(),
                    "reference_kind": "exact analytic Gaussian posterior",
                    "calibration_seeds": list(SEEDS), "arms": list(ARMS),
                    "chain_design": {"warmup": WARMUP, "draws_default": DRAWS,
                                     "draws_overrides": DRAWS_BY_ARM,
                                     "chains": CHAINS, "batch_size": BATCH}}
        save_json(OUTPUT / "invariants.json", metadata)
        print("Year invariants", {k: metadata[k] for k in
              ("precision_condition_number", "max_relative_cv2_gradient_error",
               "identity_cv2_slowest_efold_steps")}, flush=True)
        # Independent exact draws anchor the sampling resolution at the same score size.
        for seed in SEEDS:
            reference = posterior.sample(1024, generator(seed, 0, 84))
            baseline = posterior.sample(512, generator(seed, 1, 84))
            baseline_score = metric.distance(baseline, reference)
            initial = posterior.mean + 2 * generator(seed, 0, 83).normal(size=(CHAINS, model.d)) @ L.T
            for name, method, h in ARMS:
                path = OUTPUT / f"{name}_seed{seed}.json"
                if path.exists():
                    print(name, seed, "already complete", flush=True)
                    continue
                estimator = prepare_estimator(model, method, anchor)
                draws = DRAWS_BY_ARM.get(name, DRAWS)
                samples, run = run_chains(estimator, initial=initial, step_size=h,
                    batch_size=BATCH, warmup=WARMUP, draws=draws, chains=CHAINS,
                    seed=seed, preconditioner="frozen_full", frozen_matrix=inverse,
                    metric_cost=metric_cost)
                if run.failures or not np.isfinite(samples).all():
                    raise RuntimeError(f"failed {name} seed {seed}")
                diag = chain_diagnostics(samples, cost=run.costs[-1])
                flat = samples.reshape(-1, model.d)
                mean_error, cov_error = moment_errors(flat, posterior, metric)
                # Spread score draws over all chains and the full retained interval.
                # A contiguous slice of one correlated chain is a poor SW2 sample.
                score_draws = samples[:, np.linspace(0, draws - 1, 128, dtype=int)].reshape(512, model.d)
                score = metric.distance(score_draws, reference)
                row = {"arm": name, "seed": seed, "method": method, "h": h,
                       "warmup": WARMUP, "draws_per_chain": draws,
                       "diagnostics": {"max_rhat": diag["max_rhat"],
                           "min_bulk_ess": diag["min_bulk_ess"],
                           "min_tail_ess": diag["min_tail_ess"],
                           "reliable": diag["reliable"]},
                       "standardized_mean_error": mean_error,
                       "standardized_covariance_error": cov_error,
                       "standardized_sw2_512_vs_1024": score,
                       "sw2_sample_design": "128 equally spaced retained draws per chain",
                       "exact_vs_exact_sw2_512_vs_1024": baseline_score,
                       "cost": run.costs[-1], "metric_fingerprint": metric_fingerprint(metric),
                       "cost_note": "sufficient statistics, redundant CV caches, full inverse and chain included; scoring excluded"}
                save_json(path, row)
                print(name, seed, "Rhat", round(diag["max_rhat"], 4),
                      "ESS", round(diag["min_bulk_ess"]), "SW2", round(score, 4), flush=True)


if __name__ == "__main__":
    main()

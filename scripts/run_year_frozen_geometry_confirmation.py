"""Run the frozen Year ridge geometry study against its exact posterior."""
import hashlib
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


CONFIG = Path("configs/year_frozen_geometry_confirmation.json")
OUTPUT = Path("results/year_frozen_geometry_confirmation")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score_moments(x, posterior, metric):
    d = len(posterior.mean)
    mean_error = np.linalg.norm((x.mean(axis=0) - posterior.mean) @ metric.transform) / np.sqrt(d)
    cov = np.cov(x, rowvar=False, ddof=0)
    covariance_error = np.linalg.norm(metric.transform.T @
        (cov - posterior.covariance) @ metric.transform) / d
    return float(mean_error), float(covariance_error)


def main():
    config = json.loads(CONFIG.read_text())
    if config["purpose"] != "year_frozen_geometry_confirmation":
        raise ValueError("wrong Year confirmation purpose")
    seeds = config["method_seeds"]
    if len(seeds) != len(set(seeds)) or set(seeds) & set(config["inspected_sampler_seeds"]):
        raise ValueError("fresh Year seeds overlap inspected seeds")
    pilot = Path(config["problem_source"])
    problem = json.loads((pilot / "manifest.json").read_text())["config"]["problem"]
    model, data_meta = year_model(**{k: v for k, v in problem.items() if k != "kind"})
    metric = load_accuracy_metric(config["scoring_calibration"], d=model.d)
    with threadpool_limits(limits=1):
        posterior = ridge_posterior(model)
        with np.load(pilot / "geometry.npz") as data:
            np.testing.assert_allclose(posterior.mean, data["anchor"], rtol=0, atol=1e-7)
            np.testing.assert_allclose(posterior.precision, data["hessian"], rtol=1e-9, atol=1e-2)
        anchor = supplied_anchor(posterior.mean,
                                 provenance="analytic full-data ridge sufficient statistics",
                                 cost=posterior.cost)
        metric_cost = CostLedger()
        with metric_cost.measure("full_metric_inverse"):
            inverse = posterior.covariance
        metric_cost.count("linear_solves")
        with metric_cost.measure("full_metric_initial_cholesky"):
            L = np.linalg.cholesky(inverse)
        protocol = {"config": config, "config_sha256": sha256(CONFIG),
                    "geometry_sha256": sha256(pilot / "geometry.npz"),
                    "scoring_calibration_sha256": metric_fingerprint(metric),
                    "data_sha256": data_meta["source_sha256"],
                    "reference_kind": "exact analytic Gaussian posterior",
                    "cost_scope": "each arm charged full sufficient-statistic anchor, CV caches, optional full metric, and sequential four-chain transitions; scoring and data loading excluded"}
        OUTPUT.mkdir(parents=True, exist_ok=True)
        protocol_path = OUTPUT / "protocol.json"
        if protocol_path.exists():
            if json.loads(protocol_path.read_text()) != protocol:
                raise ValueError("existing Year protocol differs")
        else:
            if any(OUTPUT.iterdir()):
                raise ValueError("nonempty Year output without protocol")
            save_json(protocol_path, protocol)
        for seed in seeds:
            baseline_path = OUTPUT / f"baseline_seed{seed}.json"
            reference = posterior.sample(config["reference_size"], generator(seed, 0, 84))
            if not baseline_path.exists():
                direct_cost = posterior.cost.copy()
                with direct_cost.measure("direct_gaussian_sampling"):
                    baseline = posterior.sample(config["chains"] * config["score_draws_per_chain"],
                                                generator(seed, 1, 84))
                direct_cost.count("normal_variates", baseline.size)
                direct_cost.count("triangular_solves")
                save_json(baseline_path, {"seed": seed,
                    "exact_vs_exact_sw2": metric.distance(baseline, reference),
                    "standardized_mean_error": score_moments(baseline, posterior, metric)[0],
                    "standardized_covariance_error": score_moments(baseline, posterior, metric)[1],
                    "cost": direct_cost.snapshot(),
                    "note": "direct Gaussian draw is an analytic-target benchmark, not a stochastic-gradient method"})
            initial = (posterior.mean + config["initial_posterior_scale"] *
                generator(seed, 0, 83).normal(size=(config["chains"], model.d)) @ L.T)
            for arm in generator(seed, 0, 87).permutation(list(config["arms"])):
                arm = str(arm)
                path = OUTPUT / f"result_{arm}_seed{seed}.json"
                if path.exists():
                    print(arm, seed, "already complete", flush=True)
                    continue
                setting = config["arms"][arm]
                estimator = prepare_estimator(model, setting["method"], anchor)
                options = {}
                if setting["geometry"] == "full_hessian":
                    options = {"preconditioner": "frozen_full", "frozen_matrix": inverse,
                               "metric_cost": metric_cost}
                elif setting["geometry"] != "identity":
                    raise ValueError("unknown Year geometry")
                draws = setting["draws"]
                samples, run = run_chains(estimator, initial=initial,
                    step_size=setting["step_size"], batch_size=setting["batch_size"],
                    warmup=setting["warmup"], draws=draws, chains=config["chains"],
                    seed=seed, **options)
                if run.failures or not np.isfinite(samples).all():
                    raise RuntimeError(f"Year run failed: {arm}, {seed}")
                diag = chain_diagnostics(samples, cost=run.costs[-1])
                score_samples = samples[:, np.linspace(0, draws - 1,
                    config["score_draws_per_chain"], dtype=int)].reshape(-1, model.d)
                mean_error, covariance_error = score_moments(samples.reshape(-1, model.d), posterior, metric)
                result = {"arm": arm, "seed": seed, "setting": setting,
                    "diagnostics": {"max_rhat": diag["max_rhat"],
                        "min_bulk_ess": diag["min_bulk_ess"],
                        "min_tail_ess": diag["min_tail_ess"],
                        "reliable": diag["reliable"]},
                    "standardized_mean_error": mean_error,
                    "standardized_covariance_error": covariance_error,
                    "standardized_sw2_512_vs_1024": metric.distance(score_samples, reference),
                    "cost": run.costs[-1],
                    "scoring_calibration_sha256": metric_fingerprint(metric),
                    "sw2_sample_design": "128 equally spaced retained draws from each of four chains",
                    "scope": "diagnostic on fresh seed; identity arms expected to be budget-censored, not validated samplers"}
                save_json(path, result)
                print(arm, seed, "Rhat", round(diag["max_rhat"], 4),
                      "ESS", round(diag["min_bulk_ess"]),
                      "SW2", round(result["standardized_sw2_512_vs_1024"], 4), flush=True)


if __name__ == "__main__":
    main()

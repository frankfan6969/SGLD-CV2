"""Frozen four-cell Gaussian geometry/CV-order confirmation."""
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from calibrate_gaussian_compact_geometry import full_metric
from sgld_cv.anchors import supplied_anchor
from sgld_cv.diagnostics import chain_diagnostics
from sgld_cv.estimators import prepare_estimator
from sgld_cv.gaussian import analytic_ula_w2, make_gaussian_target
from sgld_cv.randomness import generator
from sgld_cv.samplers import run_chains
from sgld_cv.transport import make_accuracy_metric, metric_fingerprint
from sgld_cv.workflow import save_json


CONFIG = Path("configs/gaussian_compact_geometry_confirmation.json")
OUTPUT = Path("results/gaussian_compact_geometry_confirmation")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def errors(x, target, metric):
    d = len(target.mean)
    mean = np.linalg.norm((x.mean(axis=0) - target.mean) @ metric.transform) / np.sqrt(d)
    cov = np.linalg.norm(metric.transform.T @
        (np.cov(x, rowvar=False, ddof=0) - target.covariance) @ metric.transform) / d
    return float(mean), float(cov)


def main():
    config = json.loads(CONFIG.read_text())
    if config["purpose"] != "gaussian_compact_geometry_confirmation":
        raise ValueError("wrong Gaussian confirmation purpose")
    if (set(config["design_seeds"]) & set(config["inspected_design_seeds"]) or
            set(config["sampler_seeds"]) & set(config["inspected_sampler_seeds"])):
        raise ValueError("held-out Gaussian seeds overlap inspected seeds")
    protocol = {"config": config, "config_sha256": digest(CONFIG),
                "source_design_sha256": digest(config["source_design"])}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol_path = OUTPUT / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("existing Gaussian protocol differs")
    else:
        if any(OUTPUT.iterdir()):
            raise ValueError("nonempty Gaussian output without protocol")
        save_json(protocol_path, protocol)
    with threadpool_limits(limits=1):
        for d in config["dimensions"]:
            for kappa in config["target_condition_numbers"]:
                for design_seed in config["design_seeds"]:
                    target = make_gaussian_target(N=config["N"], d=d, kappa=kappa,
                                                  seed=design_seed)
                    metric = make_accuracy_metric(d, kind="standardized_sliced_w2",
                        center=target.mean, covariance=target.covariance,
                        projections=config["score_projections"],
                        seed=config["score_projection_seed"])
                    target_id = f"d{d}_k{kappa}_design{design_seed}"
                    target_path = OUTPUT / f"target_{target_id}.json"
                    if not target_path.exists():
                        Q_max = float(np.linalg.eigvalsh(target.precision)[-1])
                        h_identity = config["identity_cv2_analytic_step_fraction"] / Q_max
                        identity = analytic_ula_w2(target, h_identity,
                            config["checkpoints"]) / np.sqrt(d)
                        h = config["step_size"]
                        a = 1 - h / 2
                        full = []
                        for k in config["checkpoints"]:
                            v = h * (1 - a ** (2 * k)) / (1 - a * a)
                            full.append(float(abs(np.sqrt(v) - 1)))
                        save_json(target_path, {"target": target.metadata,
                            "data_sha256": hashlib.sha256(target.model.X.tobytes() +
                                                          target.model.y.tobytes()).hexdigest(),
                            "metric_fingerprint": metric_fingerprint(metric),
                            "checkpoints": config["checkpoints"],
                            "identity_cv2_h": h_identity,
                            "identity_cv2_exact_w2_over_sqrt_d_from_mode": identity.tolist(),
                            "full_cv2_exact_w2_over_sqrt_d_from_mode": full,
                            "analytic_scope": "exact-gradient CV2 from mode only; not a Gaussian-law claim for CV1"})
                    anchor = supplied_anchor(target.mean,
                        provenance="known exact zero Gaussian mode")
                    for sampler_seed in config["sampler_seeds"]:
                        reference = target.sample(config["reference_size"],
                            generator(sampler_seed, 0, 84))
                        baseline_path = OUTPUT / f"baseline_{target_id}_seed{sampler_seed}.json"
                        if not baseline_path.exists():
                            independent = target.sample(config["chains"] *
                                config["score_draws_per_chain"],
                                generator(sampler_seed, 1, 84))
                            save_json(baseline_path, {"d": d, "kappa": kappa,
                                "design_seed": design_seed, "sampler_seed": sampler_seed,
                                "exact_vs_exact_sw2": metric.distance(independent, reference),
                                "reference_size": len(reference),
                                "score_size": len(independent)})
                        initial = config["initial_posterior_scale"] * target.sample(
                            config["chains"], generator(sampler_seed, 0, 83))
                        for method in generator(sampler_seed, 0, 87).permutation(config["methods"]):
                            method = str(method)
                            path = OUTPUT / f"result_{target_id}_{method}_seed{sampler_seed}.json"
                            if path.exists():
                                print(target_id, method, sampler_seed, "already complete", flush=True)
                                continue
                            estimator = prepare_estimator(target.model, method, anchor)
                            P, P_cost = full_metric(target, estimator)
                            samples, run = run_chains(estimator, initial=initial,
                                step_size=config["step_size"], batch_size=config["batch_size"],
                                warmup=config["warmup"], draws=config["draws_per_chain"],
                                chains=config["chains"], seed=sampler_seed,
                                preconditioner="frozen_full", frozen_matrix=P,
                                metric_cost=P_cost)
                            if run.failures or not np.isfinite(samples).all():
                                raise RuntimeError(f"Gaussian failure: {target_id} {method} {sampler_seed}")
                            diag = chain_diagnostics(samples)
                            flat = samples.reshape(-1, d)
                            sampled = samples[:, np.linspace(0, config["draws_per_chain"] - 1,
                                config["score_draws_per_chain"], dtype=int)].reshape(-1, d)
                            mean, covariance = errors(flat, target, metric)
                            row = {"d": d, "kappa": kappa, "design_seed": design_seed,
                                "sampler_seed": sampler_seed, "method": method,
                                "step_size": config["step_size"],
                                "batch_size": config["batch_size"],
                                "max_rhat": diag["max_rhat"],
                                "min_bulk_ess": diag["min_bulk_ess"],
                                "min_tail_ess": diag["min_tail_ess"],
                                "passes_mixing_gate": bool(diag["max_rhat"] <
                                    config["mixing_gate"]["max_rhat_exclusive"] and
                                    min(diag["min_bulk_ess"], diag["min_tail_ess"]) >=
                                    config["mixing_gate"]["min_bulk_ess_inclusive"]),
                                "standardized_mean_error": mean,
                                "standardized_covariance_error": covariance,
                                "standardized_sw2_512_vs_1024": metric.distance(sampled, reference),
                                "cost": run.costs[-1],
                                "metric_fingerprint": metric_fingerprint(metric),
                                "scoring_design": "128 evenly spaced retained draws from each of four chains"}
                            save_json(path, row)
                            print(target_id, method, sampler_seed, "Rhat",
                                  round(diag["max_rhat"], 4), "SW2",
                                  round(row["standardized_sw2_512_vs_1024"], 4), flush=True)


if __name__ == "__main__":
    main()

"""Run the frozen six-method 2D logistic comparison at a fixed K=8192.

This is a held-out fixed-horizon comparison, not a K_needed certificate.
Each method/seed is committed separately so interrupted work can resume.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from sgld_cv.anchors import find_anchor
from sgld_cv.data import synthetic_logistic
from sgld_cv.estimators import prepare_estimator
from sgld_cv.preconditioners import metric_pilot
from sgld_cv.randomness import generator
from sgld_cv.samplers import run_replicas
from sgld_cv.transport import make_accuracy_metric
from sgld_cv.workflow import METHODS, save_json


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(config_path, output):
    config_path, output = Path(config_path), Path(output)
    config = json.loads(config_path.read_text())
    if config["purpose"] not in {"heldout_fixed_horizon_comparison",
                                  "heldout_postdiagnostic_confirmation"}:
        raise ValueError("incorrect purpose")
    if config["methods"] != list(METHODS):
        raise ValueError("the protocol must include all six methods")
    seeds = config["method_seeds"]
    if len(seeds) != len(set(seeds)) or set(seeds) & set(config["inspected_method_seeds"]):
        raise ValueError("held-out seeds overlap inspected seeds or each other")
    if config["steps"] != 8192 or config["checkpoints"] != [0, 2048, 4096, 8192]:
        raise ValueError("unexpected fixed-horizon protocol")
    pilot = Path(config["problem_source"])
    selection_path = Path(config["selection_source"])
    reference_folder = Path(config["reference_source"])
    reference_path = reference_folder / "reference_diagnostics.json"
    chains_path = reference_folder / "reference_chains.npz"
    selection = json.loads(selection_path.read_text())
    if "final_horizon" in selection:
        raise ValueError("this protocol must not assert a verified K_needed")
    if set(selection["selection"]) != set(METHODS):
        raise ValueError("selection does not contain six methods")
    reference_meta = json.loads(reference_path.read_text())
    if not reference_meta["reliable"]:
        raise ValueError("independent HMC reference failed validation")
    if [reference_meta["first"]["seed"], reference_meta["longer"]["seed"]] != config["reference_seeds"]:
        raise ValueError("reference seeds differ from the frozen protocol")
    pilot_config = json.loads((pilot / "manifest.json").read_text())["config"]
    model, _ = synthetic_logistic(**{k: v for k, v in pilot_config["problem"].items()
                                   if k != "kind"})
    with np.load(pilot / "geometry.npz") as data:
        theta, H = data["anchor"], data["hessian"]
    with np.load(chains_path) as data:
        first = data["first"].reshape(-1, model.d)
        longer = data["longer"].reshape(-1, model.d)
    R, S = config["replicas_per_seed"], config["reference_size"]
    if R != 768 or S != 3072 or min(len(first), len(longer)) < max(R, S):
        raise ValueError("replica/reference sizes differ from the frozen protocol")
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"config": config, "config_sha256": sha256(config_path),
                "selection_sha256": sha256(selection_path),
                "reference_diagnostics_sha256": sha256(reference_path),
                "reference_chains_sha256": sha256(chains_path),
                "pilot_geometry_sha256": sha256(pilot / "geometry.npz"),
                "status": "fixed-horizon held-out evaluation; K_needed not asserted"}
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("existing output has a different frozen protocol")
    else:
        if any(output.iterdir()):
            raise ValueError("output directory is nonempty without a matching protocol")
        save_json(protocol_path, protocol)
    metric = make_accuracy_metric(model.d, kind="w2")
    anchor = find_anchor(model)
    np.testing.assert_allclose(anchor.theta, theta, rtol=0, atol=1e-10)
    pilots = json.loads((pilot / "metric_pilots.json").read_text())
    metric_pilots = {}
    for b in {selection["selection"][m]["batch_size"] for m in METHODS if m.startswith("p")}:
        new_pilot = metric_pilot(model, anchor.theta, batch_size=b,
                                 **pilot_config["metric"])
        np.testing.assert_array_equal(new_pilot.second_moment,
                                      np.asarray(pilots[str(b)]["v"]))
        metric_pilots[b] = new_pilot
    ref_mean = np.asarray(reference_meta["reference_mean"])
    ref_cov = np.asarray(reference_meta["reference_covariance"])
    whitening = np.linalg.cholesky(H)
    with threadpool_limits(limits=1):
        for seed in seeds:
            ref = longer[generator(seed, 0, 84).permutation(len(longer))[:S]]
            independent = first[generator(seed, 1, 85).permutation(len(first))[:R]]
            baseline_path = output / f"baseline_seed{seed}.json"
            if not baseline_path.exists():
                save_json(baseline_path, {"seed": seed, "replicas": R,
                    "reference_size": S,
                    "reference_reference_w2": metric.distance(independent, ref)})
            for method in generator(seed, 0, 87).permutation(METHODS):
                method = str(method)
                row_path = output / f"result_{method}_seed{seed}.json"
                state_path = output / f"states_{method}_seed{seed}.npz"
                if row_path.exists() and state_path.exists():
                    print(method, seed, "already complete", flush=True)
                    continue
                if row_path.exists() or state_path.exists():
                    raise ValueError(f"partial result requires inspection: {method}, {seed}")
                setting = selection["selection"][method]
                b = setting["batch_size"]
                h = setting["step_size"] * config.get(
                    "step_size_multipliers", {}).get(method, 1.0)
                base = method[1:] if method.startswith("p") else method
                estimator = prepare_estimator(model, base, anchor)
                options = {}
                if method.startswith("p"):
                    options = {"preconditioner": "rmsprop",
                        "initial_v": metric_pilots[b].second_moment,
                        "metric_cost": metric_pilots[b].cost,
                        "alpha": pilot_config["metric"].get("alpha", .99),
                        "epsilon": pilot_config["metric"].get("epsilon", 1e-5)}
                trajectory = run_replicas(estimator, initial=theta,
                    step_size=h, batch_size=b, steps=config["steps"],
                    replicas=R, seed=seed, checkpoints=config["checkpoints"],
                    **options)
                if trajectory.failures or not np.isfinite(trajectory.states).all():
                    raise RuntimeError(f"{method}, seed {seed}: failed trajectories")
                checkpoints = []
                for k, x, cost in zip(trajectory.checkpoints, trajectory.states,
                                      trajectory.costs):
                    mean_error = float(np.linalg.norm((x.mean(axis=0)-ref_mean) @
                                                      whitening) / np.sqrt(model.d))
                    covariance_error = float(np.linalg.norm(whitening.T @
                        (np.cov(x, rowvar=False, ddof=0)-ref_cov) @ whitening) / model.d)
                    checkpoints.append({"iteration": int(k),
                        "w2": metric.distance(x, ref),
                        "mean_error_standardized": mean_error,
                        "covariance_error_standardized": covariance_error,
                        "mean_error_guarded": mean_error + reference_meta["standardized_mean_difference"],
                        "covariance_error_guarded": covariance_error + reference_meta["standardized_covariance_difference"],
                        "cost": cost})
                with state_path.open("wb") as file:
                    np.savez_compressed(file, checkpoints=trajectory.checkpoints,
                                        states=trajectory.states)
                save_json(row_path, {"method": method, "seed": seed,
                    "batch_size": b, "step_size": h, "replicas": R,
                    "reference_size": S, "checkpoints": checkpoints,
                    "sampler_metadata": trajectory.metadata,
                    "cost_note": "sampler preparation and transitions; reference scoring excluded"})
                print(method, seed, "done", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
        default=Path("configs/synthetic_logistic_fixed8192_heldout.json"))
    parser.add_argument("--output", type=Path,
        default=Path("results/logistic_fixed8192_heldout"))
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == "__main__":
    main()

"""Run the frozen German Credit geometry/CV-order confirmation on fresh seeds."""
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from calibrate_german_geometry_pairs import full_metric
from sgld_cv.anchors import find_anchor
from sgld_cv.data import german_credit
from sgld_cv.estimators import prepare_estimator
from sgld_cv.randomness import generator
from sgld_cv.samplers import run_replicas
from sgld_cv.transport import load_accuracy_metric, metric_fingerprint
from sgld_cv.workflow import save_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    config_path = Path("configs/german_frozen_geometry_confirmation.json")
    output = Path("results/german_frozen_geometry_confirmation")
    config = json.loads(config_path.read_text())
    if config["purpose"] != "german_frozen_geometry_confirmation":
        raise ValueError("incorrect German confirmation purpose")
    seeds = config["method_seeds"]
    if len(seeds) != len(set(seeds)) or set(seeds) & set(config["inspected_sampler_seeds"]):
        raise ValueError("German confirmation seeds overlap inspected seeds")
    if config["steps"] != 4096 or config["checkpoints"] != [0, 1024, 2048, 4096]:
        raise ValueError("changed frozen German checkpoint protocol")
    pilot = Path(config["problem_source"])
    ref_folder = Path(config["reference_source"])
    refmeta = json.loads((ref_folder / "reference_diagnostics.json").read_text())
    if not refmeta["reliable"]:
        raise ValueError("fresh German HMC reference failed validation")
    metric = load_accuracy_metric(config["scoring_calibration"], d=61)
    fingerprint = metric_fingerprint(metric)
    expected = json.loads((pilot / "frozen_selection.json").read_text())[
        "accuracy_metric"]["calibration_sha256"]
    if fingerprint != expected or refmeta["calibration_sha256"] != expected:
        raise ValueError("German scoring calibration was not frozen")
    problem = json.loads((pilot / "manifest.json").read_text())["config"]["problem"]
    model, data_meta = german_credit(**{k: v for k, v in problem.items()
                                        if k != "kind"})
    anchor = find_anchor(model)
    with np.load(pilot / "geometry.npz") as data:
        np.testing.assert_allclose(anchor.theta, data["anchor"], rtol=0, atol=1e-9)
        hessian = np.asarray(data["hessian"])
    with np.load(ref_folder / "reference_chains.npz") as data:
        first = data["first"].reshape(-1, model.d)
        longer = data["longer"].reshape(-1, model.d)
    if min(len(first), len(longer)) < max(config["replicas_per_seed"],
                                          config["reference_size"]):
        raise ValueError("German HMC pool too small")
    protocol = {"config": config, "config_sha256": digest(config_path),
                "geometry_sha256": digest(pilot / "geometry.npz"),
                "reference_diagnostics_sha256": digest(ref_folder / "reference_diagnostics.json"),
                "reference_chains_sha256": digest(ref_folder / "reference_chains.npz"),
                "scoring_calibration_sha256": fingerprint,
                "data_sha256": data_meta["source_sha256"],
                "interpretation": "frozen fresh-seed German Credit comparison; no K_needed claim"}
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("existing German confirmation protocol differs")
    else:
        if any(output.iterdir()):
            raise ValueError("nonempty German confirmation output without protocol")
        save_json(protocol_path, protocol)
    ref_mean = np.asarray(refmeta["reference_mean"])
    ref_cov = np.asarray(refmeta["reference_covariance"])
    R, S = config["replicas_per_seed"], config["reference_size"]
    with threadpool_limits(limits=1):
        for seed in seeds:
            ref = longer[generator(seed, 0, 84).permutation(len(longer))[:S]]
            independent = first[generator(seed, 1, 85).permutation(len(first))[:R]]
            baseline_path = output / f"baseline_seed{seed}.json"
            if not baseline_path.exists():
                save_json(baseline_path, {"seed": seed, "replicas": R,
                    "reference_size": S, "reference_reference_sw2": metric.distance(independent, ref)})
            for arm in generator(seed, 0, 87).permutation(list(config["arms"])):
                arm = str(arm)
                state_path = output / f"states_{arm}_seed{seed}.npz"
                result_path = output / f"result_{arm}_seed{seed}.json"
                if state_path.exists() and result_path.exists():
                    print(arm, seed, "already complete", flush=True)
                    continue
                if state_path.exists() or result_path.exists():
                    raise ValueError(f"partial German confirmation result: {arm}, {seed}")
                setting = config["arms"][arm]
                estimator = prepare_estimator(model, setting["method"], anchor)
                options = {}
                if setting["geometry"] == "full_hessian":
                    matrix, metric_cost = full_metric(model, estimator, hessian)
                    options = {"preconditioner": "frozen_full",
                               "frozen_matrix": matrix, "metric_cost": metric_cost}
                elif setting["geometry"] != "identity":
                    raise ValueError("unknown German confirmation geometry")
                run = run_replicas(estimator, initial=anchor.theta,
                    step_size=setting["step_size"], batch_size=config["batch_size"],
                    steps=config["steps"], replicas=R, seed=seed,
                    checkpoints=config["checkpoints"], **options)
                if run.failures or not np.isfinite(run.states).all():
                    raise RuntimeError(f"failed German confirmation: {arm}, {seed}")
                rows = []
                for k, x, cost in zip(run.checkpoints, run.states, run.costs):
                    mean = (np.linalg.norm((x.mean(axis=0) - ref_mean) @ metric.transform) /
                            np.sqrt(model.d))
                    cov = np.linalg.norm(metric.transform.T @
                        (np.cov(x, rowvar=False, ddof=0) - ref_cov) @
                        metric.transform) / model.d
                    rows.append({"iteration": int(k), "sw2": metric.distance(x, ref),
                        "standardized_mean_error": float(mean),
                        "standardized_covariance_error": float(cov), "cost": cost})
                with state_path.open("wb") as stream:
                    np.savez_compressed(stream, checkpoints=run.checkpoints,
                                        states=run.states)
                save_json(result_path, {"arm": arm, "method": setting["method"],
                    "geometry": setting["geometry"], "seed": seed,
                    "step_size": setting["step_size"], "batch_size": config["batch_size"],
                    "replicas": R, "reference_size": S,
                    "checkpoints": rows, "sampler_metadata": run.metadata,
                    "scoring_calibration_sha256": fingerprint,
                    "cost_note": "setup and all transitions included; HMC generation/scoring excluded"})
                print(arm, seed, "done", flush=True)


if __name__ == "__main__":
    main()

"""Run the prespecified K=16,384 CV2 half-step check on confirmation seeds."""
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from sgld_cv.anchors import find_anchor
from sgld_cv.data import synthetic_logistic
from sgld_cv.estimators import prepare_estimator
from sgld_cv.randomness import generator
from sgld_cv.samplers import run_replicas
from sgld_cv.transport import make_accuracy_metric
from sgld_cv.workflow import save_json


FOLDER = Path("results/logistic_postdiagnostic_confirmation")
CHECKPOINTS = [0, 2048, 4096, 8192, 16384]


def main():
    protocol = json.loads((FOLDER / "protocol.json").read_text())
    config = protocol["config"]
    if config["purpose"] != "heldout_postdiagnostic_confirmation":
        raise ValueError("extension requires frozen confirmation protocol")
    selection = json.loads(Path(config["selection_source"]).read_text())["selection"]
    setting = selection["cv2"]
    h = setting["step_size"] * config["step_size_multipliers"]["cv2"]
    reference_folder = Path(config["reference_source"])
    reference_meta = json.loads((reference_folder / "reference_diagnostics.json").read_text())
    if not reference_meta["reliable"]:
        raise ValueError("reference failed diagnostics")
    with np.load(reference_folder / "reference_chains.npz") as data:
        reference_pool = data["longer"].reshape(-1, 2)
    pilot = Path(config["problem_source"])
    problem = json.loads((pilot / "manifest.json").read_text())["config"]["problem"]
    model, _ = synthetic_logistic(**{k: v for k, v in problem.items() if k != "kind"})
    anchor = find_anchor(model)
    with np.load(pilot / "geometry.npz") as data:
        np.testing.assert_allclose(anchor.theta, data["anchor"], rtol=0, atol=1e-10)
        whitening = np.linalg.cholesky(data["hessian"])
    estimator = prepare_estimator(model, "cv2", anchor)
    metric = make_accuracy_metric(model.d, kind="w2")
    for seed in config["method_seeds"]:
        row_path = FOLDER / f"result_cv2_long_seed{seed}.json"
        state_path = FOLDER / f"states_cv2_long_seed{seed}.npz"
        if row_path.exists() and state_path.exists():
            print("cv2_long", seed, "already complete", flush=True)
            continue
        if row_path.exists() or state_path.exists():
            raise ValueError(f"partial CV2 extension output, seed {seed}")
        with threadpool_limits(limits=1):
            run = run_replicas(estimator, initial=anchor.theta, step_size=h,
                               batch_size=setting["batch_size"], steps=CHECKPOINTS[-1],
                               replicas=config["replicas_per_seed"], seed=seed,
                               checkpoints=CHECKPOINTS)
        if run.failures or not np.isfinite(run.states).all():
            raise RuntimeError(f"CV2 extension failure, seed {seed}")
        with np.load(FOLDER / f"states_cv2_seed{seed}.npz") as data:
            for k, x in zip(data["checkpoints"], data["states"]):
                np.testing.assert_array_equal(run.states[CHECKPOINTS.index(int(k))], x)
        ref = reference_pool[generator(seed, 0, 84).permutation(len(reference_pool))[:config["reference_size"]]]
        ref_mean = np.asarray(reference_meta["reference_mean"])
        ref_cov = np.asarray(reference_meta["reference_covariance"])
        scored = []
        with threadpool_limits(limits=1):
            for k, x, cost in zip(run.checkpoints, run.states, run.costs):
                mean = np.linalg.norm((x.mean(axis=0) - ref_mean) @ whitening) / np.sqrt(model.d)
                cov = np.linalg.norm(whitening.T @
                    (np.cov(x, rowvar=False, ddof=0) - ref_cov) @ whitening) / model.d
                scored.append({"iteration": int(k), "w2": metric.distance(x, ref),
                    "mean_error_guarded": float(mean + reference_meta["standardized_mean_difference"]),
                    "covariance_error_guarded": float(cov + reference_meta["standardized_covariance_difference"]),
                    "cost": cost})
        with state_path.open("wb") as stream:
            np.savez_compressed(stream, checkpoints=run.checkpoints, states=run.states)
        save_json(row_path, {"method": "cv2_long", "seed": seed,
            "batch_size": setting["batch_size"], "step_size": h,
            "replicas": config["replicas_per_seed"],
            "reference_size": config["reference_size"], "checkpoints": scored,
            "sampler_metadata": run.metadata,
            "cost_note": "prespecified longer CV2 half-step run; setup and transitions included, scoring excluded"})
        print("cv2_long", seed, "done", flush=True)


if __name__ == "__main__":
    main()

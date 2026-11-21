"""Post-held-out CV1 horizon and CV2 half-step replica diagnostics."""
import argparse
import hashlib
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


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(config_path, output):
    config_path, output = Path(config_path), Path(output)
    config = json.loads(config_path.read_text())
    if config["purpose"] != "post_heldout_targeted_diagnostics":
        raise ValueError("incorrect diagnostic purpose")
    formal = Path(config["source_heldout"])
    prior = json.loads((formal / "protocol.json").read_text())
    if (config["seeds"] != prior["config"]["method_seeds"] or
            config["replicas"] != prior["config"]["replicas_per_seed"] or
            config["reference_size"] != prior["config"]["reference_size"] or
            config["source_reference"] != prior["config"]["reference_source"]):
        raise ValueError("diagnostic pairing does not match held-out experiment")
    selection = json.loads(Path(config["source_selection"]).read_text())["selection"]
    reference_folder = Path(config["source_reference"])
    refmeta = json.loads((reference_folder / "reference_diagnostics.json").read_text())
    if not refmeta["reliable"]:
        raise ValueError("reference failed reliability checks")
    chains_path = reference_folder / "reference_chains.npz"
    if digest(chains_path) != prior["reference_chains_sha256"]:
        raise ValueError("reference draws differ from held-out experiment")
    with np.load(chains_path) as data:
        longer = data["longer"].reshape(-1, 2)
    pilot = Path(config["source_pilot"])
    base_config = json.loads((pilot / "manifest.json").read_text())["config"]
    model, _ = synthetic_logistic(**{k:v for k,v in base_config["problem"].items()
                                   if k != "kind"})
    anchor = find_anchor(model)
    with np.load(pilot / "geometry.npz") as data:
        np.testing.assert_allclose(anchor.theta, data["anchor"], rtol=0, atol=1e-10)
        whitening = np.linalg.cholesky(data["hessian"])
    metric = make_accuracy_metric(model.d, kind="w2")
    if output.exists():
        saved = json.loads((output / "protocol.json").read_text())
        if saved != {"config":config,"config_sha256":digest(config_path),
                     "formal_protocol_sha256":digest(formal / "protocol.json")}:
            raise ValueError("existing diagnostic output has a different protocol")
    else:
        output.mkdir(parents=True)
        save_json(output / "protocol.json", {"config":config,
            "config_sha256":digest(config_path),
            "formal_protocol_sha256":digest(formal / "protocol.json")})
    cases = [("cv2_half", "cv2", config["cv2"]),
             ("cv1_extended", "cv1", config["cv1"])]
    for label, method, spec in cases:
        setting = selection[method]
        h = setting["step_size"] * spec["step_scale"]
        checkpoints = spec["replica_checkpoints"]
        estimator = prepare_estimator(model, method, anchor)
        for seed in config["seeds"]:
            state_path = output / f"states_{label}_seed{seed}.npz"
            row_path = output / f"result_{label}_seed{seed}.json"
            if state_path.exists() and row_path.exists():
                print(label, seed, "already complete", flush=True)
                continue
            if state_path.exists() or row_path.exists():
                raise ValueError(f"partial output for {label}, seed {seed}")
            with threadpool_limits(limits=1):
                trajectory = run_replicas(estimator, initial=anchor.theta,
                    step_size=h, batch_size=setting["batch_size"],
                    steps=checkpoints[-1], replicas=config["replicas"],
                    seed=seed, checkpoints=checkpoints)
            if trajectory.failures or not np.isfinite(trajectory.states).all():
                raise RuntimeError(f"failed {label}, seed {seed}")
            if method == "cv1":
                with np.load(formal / f"states_cv1_seed{seed}.npz") as data:
                    old = {int(k):x for k,x in zip(data["checkpoints"],data["states"])}
                for index,k in enumerate(trajectory.checkpoints):
                    if int(k) in old:
                        np.testing.assert_array_equal(trajectory.states[index],old[int(k)])
            ref = longer[generator(seed,0,84).permutation(len(longer))[:config["reference_size"]]]
            scores = []
            with threadpool_limits(limits=1):
                for k,x,cost in zip(trajectory.checkpoints,trajectory.states,trajectory.costs):
                    mean_error = float(np.linalg.norm((x.mean(axis=0)-refmeta["reference_mean"]) @
                        whitening) / np.sqrt(model.d) + refmeta["standardized_mean_difference"])
                    covariance_error = float(np.linalg.norm(whitening.T @
                        (np.cov(x,rowvar=False,ddof=0)-refmeta["reference_covariance"]) @
                        whitening) / model.d + refmeta["standardized_covariance_difference"])
                    scores.append({"iteration":int(k),"w2":metric.distance(x,ref),
                        "guarded_mean_error":mean_error,
                        "guarded_covariance_error":covariance_error,
                        "cost":cost})
            with state_path.open("wb") as file:
                np.savez_compressed(file,checkpoints=trajectory.checkpoints,
                                    states=trajectory.states)
            save_json(row_path,{"label":label,"method":method,"seed":seed,
                "step_size":h,"batch_size":setting["batch_size"],
                "replicas":config["replicas"],"reference_size":config["reference_size"],
                "checkpoints":scores,"metadata":trajectory.metadata,
                "interpretation":"Exploratory extension; does not replace held-out result."})
            completed=sum((output/f"result_{case}_seed{s}.json").exists()
                          for case,_,_ in cases for s in config["seeds"])
            save_json(output / "progress.json", {"completed":completed,
                "expected":len(cases)*len(config["seeds"]),
                "current_case":label,"current_seed":seed})
            print(label,seed,"done",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,
        default=Path("configs/logistic_postheldout_diagnostics.json"))
    parser.add_argument("--output",type=Path,
        default=Path("results/logistic_postheldout_replica_sensitivity"))
    args=parser.parse_args()
    run(args.config,args.output)


if __name__ == "__main__":
    main()

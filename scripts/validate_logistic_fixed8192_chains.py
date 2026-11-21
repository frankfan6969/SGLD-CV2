"""Independent long-chain check at the frozen 2D logistic step sizes."""
import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from sgld_cv.anchors import find_anchor
from sgld_cv.data import synthetic_logistic
from sgld_cv.diagnostics import chain_diagnostics
from sgld_cv.estimators import prepare_estimator
from sgld_cv.preconditioners import metric_pilot
from sgld_cv.randomness import generator
from sgld_cv.samplers import run_chains
from sgld_cv.workflow import save_json


def run(protocol_folder, output, *, seed=2500):
    protocol_folder, output = Path(protocol_folder), Path(output)
    protocol = json.loads((protocol_folder / "protocol.json").read_text())["config"]
    selection = json.loads(Path(protocol["selection_source"]).read_text())["selection"]
    lengths = json.loads(Path("results/logistic_step_validation/validated_steps.json").read_text())["results"]
    pilot = Path(protocol["problem_source"])
    base_config = json.loads((pilot / "manifest.json").read_text())["config"]
    model, _ = synthetic_logistic(**{k: v for k, v in base_config["problem"].items()
                                   if k != "kind"})
    anchor = find_anchor(model)
    with np.load(pilot / "geometry.npz") as data:
        np.testing.assert_allclose(anchor.theta, data["anchor"], rtol=0, atol=1e-10)
        H = data["hessian"]
    reference = json.loads((Path(protocol["reference_source"]) /
                            "reference_diagnostics.json").read_text())
    if not reference["reliable"]:
        raise ValueError("reference failed validation")
    whitening = np.linalg.cholesky(H)
    transform = np.linalg.cholesky(np.linalg.inv(H))
    ref_mean = np.asarray(reference["reference_mean"])
    ref_cov = np.asarray(reference["reference_covariance"])
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    if summary_path.exists():
        raise ValueError("chain validation already complete")
    results = {}
    with threadpool_limits(limits=1):
        for method in protocol["methods"]:
            path = output / f"{method}.json"
            if path.exists():
                results[method] = json.loads(path.read_text())
                continue
            setting = selection[method]
            b, h = setting["batch_size"], setting["step_size"]
            base = method[1:] if method.startswith("p") else method
            estimator = prepare_estimator(model, base, anchor)
            options = {}
            if method.startswith("p"):
                pilot_metric = metric_pilot(model, anchor.theta, batch_size=b,
                                            **base_config["metric"])
                options = {"preconditioner": "rmsprop",
                    "initial_v": pilot_metric.second_moment,
                    "metric_cost": pilot_metric.cost,
                    "alpha": base_config["metric"].get("alpha", .99),
                    "epsilon": base_config["metric"].get("epsilon", 1e-5)}
            initial = anchor.theta + generator(seed, 0, 83).normal(size=(4,model.d)) @ transform.T
            warmup, draws = lengths[method]["warmup"], lengths[method]["draws_per_chain"]
            samples, chain_run = run_chains(estimator, initial=initial,
                step_size=h, batch_size=b, warmup=warmup, draws=draws,
                chains=4, seed=seed, **options)
            if chain_run.failures:
                raise RuntimeError(f"{method}: failed chain")
            diagnostics = chain_diagnostics(samples, cost=chain_run.costs[-1],
                                            adaptive=method.startswith("p"))
            x = samples.reshape(-1, model.d)
            mean_error = float(np.linalg.norm((x.mean(axis=0)-ref_mean) @ whitening) /
                               np.sqrt(model.d) + reference["standardized_mean_difference"])
            covariance_error = float(np.linalg.norm(whitening.T @
                (np.cov(x, rowvar=False, ddof=0)-ref_cov) @ whitening) /
                model.d + reference["standardized_covariance_difference"])
            result = {"method": method, "seed": seed, "warmup": warmup,
                "draws_per_chain": draws, "chains": 4, "step_size": h,
                "batch_size": b, "diagnostics": diagnostics,
                "guarded_mean_error": mean_error,
                "guarded_covariance_error": covariance_error,
                "adaptive_ess_exploratory": method.startswith("p"),
                "cost": chain_run.costs[-1]}
            save_json(path, result)
            results[method] = result
            print(method, "chain check done", flush=True)
    save_json(summary_path, {"seed": seed, "methods": results,
        "interpretation": "Independent long-chain numerical check at frozen step sizes. It does not prove K_needed=8192; adaptive-chain ESS is exploratory."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-folder", type=Path,
        default=Path("results/logistic_fixed8192_heldout"))
    parser.add_argument("--output", type=Path,
        default=Path("results/logistic_fixed8192_chain_validation"))
    args = parser.parse_args()
    run(args.protocol_folder, args.output)


if __name__ == "__main__":
    main()

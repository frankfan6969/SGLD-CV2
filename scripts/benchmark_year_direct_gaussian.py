"""Direct exact Gaussian cost/accuracy benchmark with 16,000 retained draws."""
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from sgld_cv.data import year_model
from sgld_cv.randomness import generator
from sgld_cv.references import ridge_posterior
from sgld_cv.transport import load_accuracy_metric, metric_fingerprint
from sgld_cv.workflow import save_json

from run_year_frozen_geometry_confirmation import score_moments


FOLDER = Path("results/year_frozen_geometry_confirmation")


def main():
    protocol = json.loads((FOLDER / "protocol.json").read_text())
    config = protocol["config"]
    pilot = Path(config["problem_source"])
    problem = json.loads((pilot / "manifest.json").read_text())["config"]["problem"]
    model, _ = year_model(**{k: v for k, v in problem.items() if k != "kind"})
    metric = load_accuracy_metric(config["scoring_calibration"], d=model.d)
    if metric_fingerprint(metric) != protocol["scoring_calibration_sha256"]:
        raise ValueError("Year score calibration changed")
    with threadpool_limits(limits=1):
        posterior = ridge_posterior(model)
        for seed in config["method_seeds"]:
            path = FOLDER / f"direct_gaussian_16000_seed{seed}.json"
            if path.exists():
                continue
            reference = posterior.sample(config["reference_size"], generator(seed, 0, 84))
            cost = posterior.cost.copy()
            with cost.measure("direct_gaussian_sampling"):
                samples = posterior.sample(16000, generator(seed, 2, 84))
            cost.count("normal_variates", samples.size)
            cost.count("triangular_solves")
            score_samples = samples[np.linspace(0, len(samples) - 1, 512, dtype=int)]
            mean, covariance = score_moments(samples, posterior, metric)
            save_json(path, {"seed": seed, "draws": len(samples),
                "scored_draws": len(score_samples),
                "exact_vs_exact_sw2": metric.distance(score_samples, reference),
                "standardized_mean_error": mean,
                "standardized_covariance_error": covariance,
                "cost": cost.snapshot(),
                "note": "same 16,000 retained draws as full-H CV2; full-data sufficient-statistic setup counted"})
            print(seed, "direct exact done", flush=True)


if __name__ == "__main__":
    main()

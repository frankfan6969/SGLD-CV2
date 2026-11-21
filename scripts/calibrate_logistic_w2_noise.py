"""Reference-to-reference empirical W2 distribution at chosen sample sizes."""
import argparse
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from sgld_cv.transport import make_accuracy_metric
from sgld_cv.workflow import save_json


def run(reference, output, *, replica_sizes=(384, 768),
        reference_sizes=(384, 3072), repeats=48):
    reference, output = Path(reference), Path(output)
    with np.load(reference) as data:
        first = data["first"].reshape(-1, data["first"].shape[-1])
        longer = data["longer"].reshape(-1, data["longer"].shape[-1])
    if max(replica_sizes) > len(first) or max(reference_sizes) > len(longer):
        raise ValueError("sample size exceeds HMC pool")
    metric = make_accuracy_metric(first.shape[1], kind="w2")
    rng = np.random.default_rng(7502)
    results = {}
    with threadpool_limits(limits=1):
        for r in replica_sizes:
            for s in reference_sizes:
                scores = []
                for _ in range(repeats):
                    x = first[rng.choice(len(first), size=r, replace=False)]
                    y = longer[rng.choice(len(longer), size=s, replace=False)]
                    scores.append(metric.distance(x, y))
                results[f"R{r}_S{s}"] = {"median": float(np.median(scores)),
                    "q10": float(np.quantile(scores, .1)),
                    "q90": float(np.quantile(scores, .9)),
                    "mean": float(np.mean(scores))}
    save_json(output, {"source_reference": str(reference),
        "replica_sizes": replica_sizes, "reference_sizes": reference_sizes,
        "repeats": repeats, "results": results,
        "interpretation": "Independent-HMC empirical-W2 score distribution under matching posterior laws. Finite MCMC correlation may remain; these are diagnostic noise floors, not deterministic lower bounds."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path,
        default=Path("results/logistic_reference_refined/reference_chains.npz"))
    parser.add_argument("--output", type=Path,
        default=Path("results/logistic_reference_refined/w2_noise_baseline.json"))
    args = parser.parse_args()
    run(args.reference, args.output)


if __name__ == "__main__":
    main()

"""Rescore saved German endpoints against both independent HMC pools."""
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from sgld_cv.randomness import generator
from sgld_cv.transport import load_accuracy_metric, metric_fingerprint
from sgld_cv.workflow import save_json


FOLDER = Path("results/german_frozen_geometry_confirmation")


def main():
    summary = json.loads((FOLDER / "summary.json").read_text())
    protocol = summary["protocol"]
    config = protocol["config"]
    metric = load_accuracy_metric(config["scoring_calibration"], d=61)
    if metric_fingerprint(metric) != protocol["scoring_calibration_sha256"]:
        raise ValueError("changed German scoring calibration")
    with np.load(Path(config["reference_source"]) / "reference_chains.npz") as data:
        pools = {name: data[name].reshape(-1, 61) for name in ("first", "longer")}
    scores = {}
    with threadpool_limits(limits=1):
        for pool_name, pool in pools.items():
            scores[pool_name] = {}
            for arm in config["arms"]:
                values = []
                for seed in config["method_seeds"]:
                    with np.load(FOLDER / f"states_{arm}_seed{seed}.npz") as data:
                        endpoint = data["states"][-1]
                    reference = pool[generator(seed, 0, 84).permutation(
                        len(pool))[:config["reference_size"]]]
                    score = metric.distance(endpoint, reference)
                    if pool_name == "longer":
                        saved = json.loads((FOLDER / f"result_{arm}_seed{seed}.json").read_text())[
                            "checkpoints"][-1]["sw2"]
                        if score != saved:
                            raise ValueError("longer HMC pool did not reproduce formal score")
                    values.append(score)
                scores[pool_name][arm] = values
    contrasts = {}
    for first, second in config["primary_contrasts"]:
        name = f"{first}_minus_{second}"
        contrasts[name] = {}
        for pool_name in pools:
            values = np.asarray(scores[pool_name][first]) - np.asarray(
                scores[pool_name][second])
            contrasts[name][pool_name] = {"mean": float(values.mean()),
                "seed_differences": values,
                "negative_seed_count": int(np.sum(values < 0)),
                "positive_seed_count": int(np.sum(values > 0))}
    save_json(FOLDER / "reference_sensitivity.json", {
        "source": str(Path(config["reference_source"]) / "reference_chains.npz"),
        "scores": scores, "contrasts": contrasts,
        "interpretation": "Post hoc reference-pool sensitivity, not a second independent sampler-seed experiment; the two HMC pools were generated independently and validated together."})
    lines = ["# German reference-pool sensitivity", "",
        "Saved fresh-seed sampler endpoints were rescored with the two independently seeded, validated HMC pools. The longer pool exactly reproduces the formal scores; the first pool is a post hoc reference sensitivity check.", "",
        "| Contrast | Longer HMC pool mean difference | First HMC pool mean difference | Sign across eight seeds (longer / first) |",
        "| --- | ---: | ---: | --- |"]
    for name, values in contrasts.items():
        a, b = values["longer"], values["first"]
        lines.append(f"| {name} | {a['mean']:+.4f} | {b['mean']:+.4f} | {a['negative_seed_count']} negative / {b['negative_seed_count']} negative |")
    lines += ["", "This check does not remove HMC approximation error or make small contrasts practically meaningful; it only tests whether the signs depend on which validated HMC pool supplied the scored draws.", ""]
    (FOLDER / "REFERENCE_SENSITIVITY.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()

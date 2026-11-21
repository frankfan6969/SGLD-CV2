"""Extend CV1 full-metric h=0.1 diagnostics with exact prefix reproduction."""
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from calibrate_german_geometry_pairs import full_metric
from sgld_cv.anchors import find_anchor
from sgld_cv.data import german_credit
from sgld_cv.diagnostics import chain_diagnostics
from sgld_cv.estimators import prepare_estimator
from sgld_cv.randomness import generator
from sgld_cv.samplers import run_chains
from sgld_cv.workflow import save_json


PILOT = Path("results/german_sliced_pilot")
OUTPUT = Path("results/german_targeted_calibration")
SEEDS = (110, 111)
PREFIXES = (4000, 8000, 12000)


def main():
    config = json.loads((PILOT / "manifest.json").read_text())["config"]
    model, _ = german_credit(**{k: v for k, v in config["problem"].items()
                              if k != "kind"})
    anchor = find_anchor(model)
    with np.load(PILOT / "geometry.npz") as data:
        np.testing.assert_allclose(anchor.theta, data["anchor"], rtol=0, atol=1e-9)
        hessian = data["hessian"]
    inverse = np.linalg.solve(hessian, np.eye(model.d))
    cholesky = np.linalg.cholesky((inverse + inverse.T) / 2)
    for seed in SEEDS:
        path = OUTPUT / f"full_hessian_cv1_h0.1_extended_seed{seed}.json"
        raw = OUTPUT / f"full_hessian_cv1_h0.1_extended_seed{seed}.npz"
        if path.exists() and raw.exists():
            print(seed, "already complete", flush=True)
            continue
        if path.exists() or raw.exists():
            raise ValueError("partial CV1 chain-extension output")
        initial = anchor.theta + 2 * generator(seed, 0, 83).normal(
            size=(4, model.d)) @ cholesky.T
        with threadpool_limits(limits=1):
            estimator = prepare_estimator(model, "cv1", anchor)
            metric, metric_cost = full_metric(model, estimator, hessian)
            samples, run = run_chains(estimator, initial=initial,
                step_size=0.1, batch_size=32, warmup=1000, draws=PREFIXES[-1],
                chains=4, seed=seed, preconditioner="frozen_full",
                frozen_matrix=metric, metric_cost=metric_cost)
        if run.failures or not np.isfinite(samples).all():
            raise RuntimeError(f"failed extended CV1 chain diagnostic, seed {seed}")
        with np.load(OUTPUT / f"full_hessian_cv1_h0.1_seed{seed}.npz") as data:
            np.testing.assert_array_equal(samples[:, :4000], data["samples"])
            np.testing.assert_array_equal(initial, data["initial"])
        diagnostics = {str(k): chain_diagnostics(samples[:, :k])
                       for k in PREFIXES}
        with raw.open("wb") as stream:
            np.savez_compressed(stream, samples=samples, initial=initial)
        save_json(path, {"seed": seed, "method": "cv1", "geometry": "full_hessian",
            "h": 0.1, "warmup": 1000, "batch_size": 32,
            "retained_prefixes": diagnostics,
            "cost_at_12000": run.costs[-1],
            "prefix_reproduced": True,
            "interpretation": "exploratory independent-chain length diagnostic"})
        print(seed, [(k, round(v["max_rhat"], 4),
                      round(v["min_bulk_ess"])) for k, v in diagnostics.items()], flush=True)
    lines = ["# German Credit full-metric CV1 h=0.1 chain extension", "",
        "Four overdispersed chains per calibration seed, 1,000 warmup, unchanged h and batch=32. The 4,000-draw prefix exactly reproduces the earlier saved run.", "",
        "| Seed | Retained draws/chain | Max R-hat | Min bulk ESS | Min tail ESS |",
        "| ---: | ---: | ---: | ---: | ---: |"]
    for seed in SEEDS:
        result = json.loads((OUTPUT / f"full_hessian_cv1_h0.1_extended_seed{seed}.json").read_text())
        for k in PREFIXES:
            d = result["retained_prefixes"][str(k)]
            lines.append(f"| {seed} | {k:,} | {d['max_rhat']:.4f} | {d['min_bulk_ess']:.0f} | {d['min_tail_ess']:.0f} |")
    lines += ["", "This assesses multichain mixing at a different cost from independent-replica endpoint accuracy. It does not identify the minimum K needed or turn the two calibration seeds into held-out evidence.", ""]
    (OUTPUT / "CV1_CHAIN_EXTENSION.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()

"""Independent German Credit HMC scoring reference for the frozen-scale study."""
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from sgld_cv.anchors import find_anchor
from sgld_cv.data import german_credit
from sgld_cv.references import hmc_reference, validate_reference_pair
from sgld_cv.transport import load_accuracy_metric, metric_fingerprint
from sgld_cv.workflow import save_json


PILOT = Path("results/german_sliced_pilot")
OUTPUT = Path("results/german_confirmation_reference")
SEED = 7400
WARMUP = 1000
FIRST_DRAWS = 3000


def main():
    config = json.loads((PILOT / "manifest.json").read_text())["config"]
    model, data = german_credit(**{k: v for k, v in config["problem"].items()
                                 if k != "kind"})
    anchor = find_anchor(model)
    with np.load(PILOT / "geometry.npz") as saved:
        np.testing.assert_allclose(anchor.theta, saved["anchor"], rtol=0, atol=1e-9)
    metric = load_accuracy_metric(PILOT / "accuracy_calibration.npz", d=model.d)
    fingerprint = metric_fingerprint(metric)
    expected = json.loads((PILOT / "frozen_selection.json").read_text())[
        "accuracy_metric"]["calibration_sha256"]
    if fingerprint != expected:
        raise ValueError("German frozen scoring calibration differs")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol = {"purpose": "fresh_scoring_reference", "first_seed": SEED,
        "longer_seed": SEED + 1, "chains": 4, "warmup": WARMUP,
        "first_draws_per_chain": FIRST_DRAWS,
        "longer_draws_per_chain": 2 * FIRST_DRAWS,
        "step_size": 0.3, "leapfrog_steps": 12,
        "data_sha256": data["source_sha256"],
        "frozen_calibration_sha256": fingerprint}
    protocol_path = OUTPUT / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("existing reference protocol differs")
    else:
        save_json(protocol_path, protocol)
    chains_path = OUTPUT / "reference_chains.npz"
    diagnostics_path = OUTPUT / "reference_diagnostics.json"
    if chains_path.exists() and diagnostics_path.exists():
        print("German confirmation reference already complete", flush=True)
        return
    if chains_path.exists() or diagnostics_path.exists():
        raise ValueError("partial German confirmation reference")
    with threadpool_limits(limits=1):
        first, first_diag = hmc_reference(model, anchor, chains=4,
            warmup=WARMUP, draws=FIRST_DRAWS, seed=SEED,
            step_size=0.3, leapfrog_steps=12)
        longer, longer_diag = hmc_reference(model, anchor, chains=4,
            warmup=WARMUP, draws=2 * FIRST_DRAWS, seed=SEED + 1,
            step_size=0.3, leapfrog_steps=12)
    agreement = validate_reference_pair(first, longer, first_diag, longer_diag)
    meta = {"kind": "approximate full-data HMC",
        "first": first_diag, "longer": longer_diag,
        "agreement": agreement, "reliable": agreement["reliable"],
        "reference_mean": longer.mean(axis=(0, 1)),
        "reference_covariance": np.cov(longer.reshape(-1, model.d), rowvar=False),
        "calibration_sha256": fingerprint,
        "interpretation": "fresh HMC reference with previously frozen pilot whitening and projections"}
    with chains_path.open("wb") as stream:
        np.savez_compressed(stream, first=first, longer=longer)
    save_json(diagnostics_path, meta)
    print("reference reliable", agreement["reliable"],
          "max Rhat", longer_diag["max_rhat"],
          "min bulk ESS", longer_diag["min_bulk_ess"],
          "max mean z", agreement["max_mean_mcse_z"], flush=True)
    if not agreement["reliable"]:
        raise RuntimeError("fresh German HMC reference failed validation")


if __name__ == "__main__":
    main()

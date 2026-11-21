"""End-to-end saved-output and cost-accounting regression on a tiny grid."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def test_experiment_outputs_preserve_pairing_and_costs(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("gaussian_experiment", root / "experiments/gaussian.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = {"dimensions": [2, 3], "kappas": [2, 10], "N": 12, "batch_size": 3,
              "steps": 3, "replicas": 8, "reference_size": 10, "baseline_repeats": 1,
              "checkpoint_count": 3, "seeds": [300, 301], "dataset_seed": 221,
              "step_fraction": .1, "methods": ["sgld", "cv1", "cv2", "ula"]}
    out = tmp_path / "run"
    module.run_experiment(config, out)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert len(manifest["targets"]) == 4
    with (out / "metrics.csv").open() as f:
        metrics = list(csv.DictReader(f))
    assert all(r["status"] == "ok" for r in metrics)
    assert json.loads((out / "failures.json").read_text()) == []
    for d in config["dimensions"]:
        for k in config["kappas"]:
            for seed in config["seeds"]:
                cv2 = np.load(out / f"d{d}_k{k}_seed{seed}_cv2.npz")
                ula = np.load(out / f"d{d}_k{k}_seed{seed}_ula.npz")
                np.testing.assert_allclose(cv2["states"], ula["states"], atol=1e-12)
                cost = json.loads((out / f"d{d}_k{k}_seed{seed}_cv2_costs.json").read_text())["costs"][-1]
                assert cost["operations"]["observation_gradient_evaluations"] == 12 + 8 * 3 * 3
                assert cost["operations"]["observation_hessian_evaluations"] == 12
    with pytest.raises(FileExistsError):
        module.run_experiment(config, out)

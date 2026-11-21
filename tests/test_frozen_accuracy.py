"""Regression coverage for a portable, unchanged pilot scoring scale."""
import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from threadpoolctl import threadpool_limits

from sgld_cv import workflow, horizon
from sgld_cv.transport import load_accuracy_metric, metric_fingerprint
from test_stages456 import tiny_config


@pytest.fixture
def calibration(tmp_path):
    config = {"accuracy_metric": {"kind": "standardized_sliced_w2", "projections": 32}}
    model = SimpleNamespace(d=3)
    anchor = SimpleNamespace(theta=np.zeros(3))
    pilot = tmp_path / "pilot"
    pilot.mkdir()
    metric, metadata = workflow.prepare_accuracy_metric(
        config, pilot, model, anchor, np.eye(3),
        {"calibration_mean": [1., 2., 3.], "calibration_covariance": np.diag([4., 2., 1.])})
    selection = pilot / "frozen_selection.json"
    workflow.save_json(selection, {"accuracy_metric": metadata})
    out = tmp_path / "heldout"
    out.mkdir()
    return config, model, anchor, metric, selection, out


def restore(calibration):
    config, model, anchor, _, selection, out = calibration
    # These unusable new calibration statistics must never be consulted.
    return workflow.prepare_accuracy_metric(config, out, model, anchor, np.eye(model.d),
        {"calibration_mean": np.full(model.d, np.nan), "calibration_covariance": None},
        frozen_selection=selection)


def test_frozen_transform_and_scores_are_exact_and_not_reestimated(calibration, monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("frozen scoring must not estimate another accuracy metric")
    monkeypatch.setattr(workflow, "make_accuracy_metric", fail)
    _, _, _, pilot, _, out = calibration
    restored, metadata = restore(calibration)
    rng = np.random.default_rng(14)
    x, y = rng.normal(size=(11, 3)), rng.normal(size=(17, 3))
    assert_array_equal((x - pilot.center) @ pilot.transform @ pilot.projections,
                       (x - restored.center) @ restored.transform @ restored.projections)
    assert pilot.distance(x, y) == restored.distance(x, y)
    assert pilot.covariance_floor == restored.covariance_floor
    assert metadata["calibration"] == "frozen pilot calibration"
    assert metric_fingerprint(load_accuracy_metric(out / "accuracy_calibration.npz", d=3)) == metric_fingerprint(pilot)


@pytest.mark.parametrize("field", ["center", "transform", "projections"])
def test_modified_calibration_rejected_before_scoring(calibration, field):
    path = calibration[4].with_name("accuracy_calibration.npz")
    with np.load(path) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    if field == "projections":
        arrays[field][:, 0] *= -1  # Remains a unit vector, but changes the frozen artifact.
    else:
        arrays[field].flat[0] += .5
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="fingerprint"):
        restore(calibration)


@pytest.mark.parametrize("key,value", [("kind", "w2"), ("projections", 64),
                                      ("seed", 12), ("eigenvalue_floor", 1e-6)])
def test_changed_scoring_settings_rejected(calibration, key, value):
    calibration[0]["accuracy_metric"][key] = value
    with pytest.raises(ValueError, match="different accuracy metric"):
        restore(calibration)


def test_missing_or_unhashed_calibration_has_no_fallback(calibration):
    selection = calibration[4]
    selection.with_name("accuracy_calibration.npz").unlink()
    with pytest.raises(FileNotFoundError):
        restore(calibration)
    frozen = json.loads(selection.read_text())
    del frozen["accuracy_metric"]["calibration_sha256"]
    workflow.save_json(selection, frozen)
    with pytest.raises(ValueError, match="hashed"):
        restore(calibration)


def assert_saved_scores(folder, seed, d):
    metric = load_accuracy_metric(folder / "accuracy_calibration.npz", d=d)
    reference = np.load(folder / f"reference_seed{seed}.npy")
    with (folder / "metrics.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert {row["method"] for row in rows} == set(workflow.METHODS)
    for method in workflow.METHODS:
        with np.load(folder / f"replicas_{method}_seed{seed}.npz") as saved:
            states = dict(zip(saved["checkpoints"], saved["states"]))
        for row in (r for r in rows if r["method"] == method):
            assert row["status"] == "ok"
            assert float(row["accuracy"]) == metric.distance(states[int(row["iteration"])], reference)


def test_publication_uses_pilot_calibration_and_saved_scores(tmp_path, monkeypatch):
    config = tiny_config()
    config["accuracy_metric"] = {"kind": "standardized_sliced_w2", "projections": 32}
    pilot = tmp_path / "pilot"
    workflow.run_study(config, pilot)
    protocol = json.loads((pilot / "frozen_selection.json").read_text())
    protocol.update(final_horizon=4, horizon_probe={"seeds": [120]})
    workflow.save_json(pilot / "frozen_protocol.json", protocol)
    config["purpose"] = "publication"
    config["evaluation"]["seeds"] = [400]
    def fail(*args, **kwargs):
        pytest.fail("publication must load the pilot metric directly")
    monkeypatch.setattr(workflow, "make_accuracy_metric", fail)
    heldout = tmp_path / "heldout"
    workflow.run_study(config, heldout, frozen_selection=pilot / "frozen_protocol.json")
    manifest = json.loads((heldout / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["accuracy_metric"]["calibration_sha256"] == protocol["accuracy_metric"]["calibration_sha256"]
    assert_saved_scores(heldout, 400, 2)
    config["evaluation"]["seeds"] = [120]
    with pytest.raises(ValueError, match="horizon probe seeds"):
        workflow.run_study(config, tmp_path / "overlap", frozen_selection=pilot / "frozen_protocol.json")


def test_post_tuning_probe_preserves_portable_calibration(tmp_path, monkeypatch):
    config = tiny_config()
    config["problem"] = {"kind": "synthetic_logistic", "N": 30, "d": 2,
                         "design_kappa": 3, "seed": 1, "prior_precision": 1.}
    config["accuracy_metric"] = {"kind": "standardized_sliced_w2", "projections": 32}
    config["reference"] = {"chains": 4, "warmup": 60, "draws": 120, "seed": 7000,
                           "step_size": .3, "leapfrog_steps": 8}
    config["horizon_probe"] = {"seeds": [120], "batch_size": 4, "step_fraction": .1,
        "max_steps": 4, "checkpoints": [0, 1, 2, 3, 4], "replicas": 64, "reference_size": 64}
    pilot, probe = tmp_path / "pilot", tmp_path / "probe"
    workflow.run_study(config, pilot)
    # Exercise the protocol-emission branch, not scientific plateau validation.
    real_build = horizon.build_problem
    def build(*args):
        result = real_build(*args)
        result[5]["reliable"] = True
        result[5]["calibration_covariance"] = None
        return result
    monkeypatch.setattr(horizon, "build_problem", build)
    monkeypatch.setattr(horizon, "provisional_horizon", lambda *args, **kwargs: 2)
    config["reference"]["seed"] = 7100
    with threadpool_limits(limits=1):
        result = horizon.run_horizon_probe(config, probe, frozen_selection=pilot / "frozen_selection.json")
    assert result["phase"] == "post_tuning"
    protocol = json.loads((probe / "frozen_protocol.json").read_text())
    frozen = json.loads((pilot / "frozen_selection.json").read_text())
    assert protocol["accuracy_metric"]["calibration_sha256"] == frozen["accuracy_metric"]["calibration_sha256"]
    # Load using only the probe directory: no dependence on the pilot's location.
    metric, metadata = workflow.prepare_accuracy_metric(config, tmp_path, SimpleNamespace(d=2),
        SimpleNamespace(theta=np.zeros(2)), np.eye(2), {}, frozen_selection=probe / "frozen_protocol.json")
    assert metric_fingerprint(metric) == frozen["accuracy_metric"]["calibration_sha256"]
    assert metadata["calibration"] == "frozen pilot calibration"

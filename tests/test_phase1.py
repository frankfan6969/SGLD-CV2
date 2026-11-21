import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from sgld_cv.anchors import supplied_anchor, find_anchor
from sgld_cv.costs import CostLedger
from sgld_cv.diagnostics import gradient_audit
from sgld_cv.estimators import prepare_estimator
from sgld_cv.functionals import FunctionalSpec, functional_rmse
from sgld_cv.gaussian import make_gaussian_target
from sgld_cv.models import LogisticRegression
from sgld_cv.preconditioners import full_hessian_metric
from sgld_cv.protocol import validate_protocol, verify_lock, sha256
from sgld_cv.samplers import run_replicas


def fixture():
    rng = np.random.default_rng(3)
    model = LogisticRegression(rng.normal(size=(23, 3)), rng.integers(0, 2, 23))
    anchor = find_anchor(model)
    return model, anchor


@pytest.mark.parametrize("method", ["sgld", "cv1", "cv2", "ula"])
def test_phase_operation_partition_and_shared_metric(method):
    model, anchor = fixture()
    estimator = prepare_estimator(model, method, anchor)
    P, cost = full_hessian_metric(model, anchor, estimator)
    assert_allclose(P, np.linalg.inv(model.full_hessian(anchor.theta)), atol=1e-12)
    # CV2 reuses its Hessian; other arms charge actual metric construction.
    assert cost.operations.get("observation_hessian_evaluations", 0) == (0 if method == "cv2" else model.N)
    run = run_replicas(estimator, initial=anchor.theta, step_size=.01, batch_size=4,
                       steps=9, replicas=3, seed=2, burn_in=3, checkpoints=[0, 3, 9],
                       preconditioner="frozen_full", frozen_matrix=P, metric_cost=cost)
    for snapshot in run.costs:
        for key, total in snapshot["operations"].items():
            assert sum(p.get(key, 0) for p in snapshot["phase_operations"].values()) == total
    end = run.costs[-1]
    assert end["phase_operations"]["burn_in"]["transitions"] == 9
    assert end["phase_operations"]["sampling"]["transitions"] == 18
    assert end["operations"]["cholesky_factorizations"] == 1
    assert "burn_in" not in estimator.setup_cost.phase_operations


def test_functional_averages_match_full_trajectory_without_changing_rng():
    model, anchor = fixture()
    f = FunctionalSpec.from_geometry(anchor.theta, model.full_hessian(anchor.theta), model.X[:2])
    estimator = prepare_estimator(model, "cv2", anchor)
    kwargs = dict(initial=anchor.theta, step_size=.001, batch_size=4, steps=10,
                  replicas=4, seed=31, checkpoints=list(range(11)))
    plain = run_replicas(estimator, **kwargs)
    online = run_replicas(estimator, functionals=f, functional_burn_in=2, **kwargs)
    assert_array_equal(plain.states, online.states)
    assert plain.costs[-1]["operations"] == online.costs[-1]["operations"]
    assert np.isnan(online.functional_averages[:3]).all()
    for k in range(3, 11):
        assert_allclose(online.functional_averages[k], f(plain.states[3:k+1]).mean(axis=0))
    assert online.metadata["functional_evaluation_seconds_excluded"] > 0


def test_variance_functionals_and_rmse_bias_variance_decomposition():
    f = FunctionalSpec.from_geometry(np.zeros(2), np.eye(2))
    draws = np.array([[1., -2.], [3., 4.], [-1., 1.]])
    derived, names = f.derive(f(draws).mean(axis=0))
    assert_allclose(derived[2:4], draws.var(axis=0, ddof=0))
    assert names[2] == "coordinate_variance_0"
    estimates = np.array([[1., 3.], [2., 5.], [4., 6.]])
    truth, scales = np.array([2., 4.]), np.array([1., 2.])
    rows = functional_rmse(estimates, truth, scales, ["a", "b"])
    for j, row in enumerate(rows):
        assert_allclose(row["rmse"]**2, (estimates[:, j].mean()-truth[j])**2+estimates[:, j].var())
        assert_allclose(row["normalized_rmse"], row["rmse"]/scales[j])
    estimates[0, 0] = np.nan
    assert all(row["status"] == "failed_ensemble" for row in functional_rmse(estimates, truth, scales, ["a", "b"]))


def test_frozen_radius_failure_is_retained_and_stops_replica():
    target = make_gaussian_target(N=20, d=2, kappa=3, seed=4)
    est = prepare_estimator(target.model, "sgld")
    run = run_replicas(est, initial=np.zeros(2), step_size=1., batch_size=3,
                       steps=5, replicas=3, seed=1, checkpoints=[0, 1, 5],
                       failure_transform=np.eye(2), failure_center=np.zeros(2), max_standardized_radius=1e-12)
    assert len(run.failures) == 3
    assert all(f["iteration"] == 1 and f["reason"] == "frozen radius limit" for f in run.failures)
    assert np.isnan(run.states[1:]).all()
    assert run.costs[-1]["operations"]["transitions"] == 3


def test_metric_gradient_mse_scales_with_metric():
    model, anchor = fixture()
    estimators = {m: prepare_estimator(model, m, anchor) for m in ("sgld", "cv1", "cv2", "ula")}
    rows, errors = gradient_audit(estimators, [anchor.theta+.1, anchor.theta-.1], batch_size=4,
                                  batches=8, seed=1, metric_matrix=3*np.eye(model.d))
    for row in rows:
        assert_allclose(row["mean_preconditioned_mse"], 3*errors[row["method"]].mean(), atol=1e-12)
    assert errors["ula"].max() == 0


def test_adaptive_sgld_reuses_raw_aggregation():
    model, anchor = fixture()
    est = prepare_estimator(model, "sgld", anchor)
    cost = CostLedger()
    gradient, raw = est.gradient(anchor.theta+.1, np.arange(4), cost, return_raw_mean=True)
    assert_allclose(gradient, model.N*raw+model.prior_gradient(anchor.theta+.1))
    assert cost.operations["metric_batch_gradient_aggregations"] == 1
    assert cost.operations.get("batch_gradient_aggregations", 0) == 0
    assert cost.operations["observation_gradient_evaluations"] == 4


def test_protocol_rejects_seed_leakage_and_missing_controls():
    path = Path(__file__).resolve().parents[1]/"configs/phase1_protocol.json"
    config = json.loads(path.read_text())
    validate_protocol(config)
    config["seeds"]["heldout"][0] = config["seeds"]["selection"][0]
    with pytest.raises(ValueError, match="seed"):
        validate_protocol(config)
    config = json.loads(path.read_text())
    del config["arms"]["full_ula"]
    with pytest.raises(ValueError, match="incomplete"):
        validate_protocol(config)


def test_protocol_lock_detects_modified_asset(tmp_path):
    asset = tmp_path/"asset.json"
    asset.write_text("original")
    lock = {"files": {"asset.json": sha256(asset)}}
    verify_lock(lock, tmp_path)
    asset.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        verify_lock(lock, tmp_path)

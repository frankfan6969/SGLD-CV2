import json
from itertools import combinations
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from sgld_cv.anchors import supplied_anchor
from sgld_cv.estimators import prepare_estimator
from sgld_cv.models import LogisticRegression, GaussianRegression
from sgld_cv.preconditioners import fixed_anchor_metric
from sgld_cv.protocol import validate_protocol
from sgld_cv.samplers import run_replicas, run_with_theory_step
from sgld_cv.uncertainty import replica_bootstrap, paired_bootstrap, hierarchical_paired_bootstrap, cost_views


def fixture():
    rng = np.random.default_rng(4)
    model = LogisticRegression(rng.normal(size=(7, 3)), [0, 1, 0, 1, 1, 0, 1], .7)
    anchor = supplied_anchor([.1, -.2, .3], provenance="algebra anchor")
    return model, anchor


@pytest.mark.parametrize("kind", ["full", "diagonal"])
@pytest.mark.parametrize("method", ["cv1", "cv2", "ula"])
def test_fixed_metric_ridge_and_estimator_parity(kind, method):
    model, anchor = fixture()
    est = prepare_estimator(model, method, anchor)
    theta, batch = anchor.theta+.2, np.arange(3)
    before = est.gradient(theta, batch)
    H = model.full_hessian(anchor.theta)
    P, cost, meta = fixed_anchor_metric(model, anchor, est, kind=kind)
    delta = 1e-10*model.prior_precision
    expected = np.linalg.inv(H+delta*np.eye(model.d)) if kind == "full" else 1/(np.diag(H)+delta)
    assert_allclose(P, expected, atol=1e-13)
    assert_array_equal(est.gradient(theta, batch), before)
    assert meta["ridge_shift"] == delta
    if method == "cv2":
        assert est.hessian_hat.shape == (model.d, model.d)
        assert cost.operations.get("observation_hessian_evaluations", 0) == 0
        assert_allclose(est.hessian_hat, H-model.prior_precision*np.eye(model.d), atol=1e-15)
    elif kind == "diagonal":
        assert cost.operations["diagonal_hessian_assemblies"] == 1
        assert cost.operations.get("dense_hessian_assemblies", 0) == 0


def test_ridge_is_not_a_silent_spd_repair():
    model, anchor = fixture()
    est = prepare_estimator(model, "cv2", anchor)
    est.hessian_hat = -100*np.eye(model.d)
    with pytest.raises(np.linalg.LinAlgError):
        fixed_anchor_metric(model, anchor, est, kind="full")
    with pytest.raises(ValueError, match="positive"):
        fixed_anchor_metric(model, anchor, est, kind="diagonal")
    with pytest.raises(ValueError, match="nonnegative"):
        fixed_anchor_metric(model, anchor, relative_ridge=-1)


def test_public_step_adapter_preserves_legacy_trajectory_and_cost():
    model, anchor = fixture()
    est = prepare_estimator(model, "cv2", anchor)
    opts = dict(initial=anchor.theta, batch_size=3, steps=8, replicas=4, seed=12,
                preconditioner="rmsprop", record_adaptive_state=True)
    public = run_with_theory_step(est, h=1e-6, **opts)
    legacy = run_replicas(est, step_size=2e-6, **opts)
    assert_array_equal(public.states, legacy.states)
    assert_array_equal(public.adaptive_final_v, legacy.adaptive_final_v)
    assert public.costs[-1]["operations"] == legacy.costs[-1]["operations"]
    assert public.costs[-1]["phase_operations"]["adaptive_state_initialization"]["adaptive_state_initializations"] == 4
    with pytest.raises(ValueError, match="never both"):
        run_with_theory_step(est, h=1e-6, step_size=2e-6, **opts)


def test_profile_components_are_nested_and_do_not_change_randomness():
    model, anchor = fixture()
    est = prepare_estimator(model, "cv1", anchor)
    opts = dict(initial=anchor.theta, h=1e-6, batch_size=3, steps=8, replicas=3, seed=12,
                preconditioner="rmsprop", record_adaptive_state=True)
    a = run_with_theory_step(est, profile_components=True, **opts)
    b = run_with_theory_step(est, **opts)
    assert_array_equal(a.states, b.states)
    assert_array_equal(a.adaptive_final_v, b.adaptive_final_v)
    final = a.costs[-1]
    assert sum(final["component_seconds"].values()) <= final["seconds"]["sampling"]
    assert_allclose(final["total_seconds"], sum(final["seconds"].values()))
    views = cost_views(final, method_tuning_seconds=3.5)
    assert_allclose(views["search_inclusive_total_seconds"], final["total_seconds"]+3.5)


@pytest.mark.parametrize("n", [1, 2, 4, 7])
def test_exact_finite_population_variance_and_quartic_bound(n):
    model, anchor = fixture()
    theta = anchor.theta+np.array([.4, -.2, .3])
    est = prepare_estimator(model, "cv2", anchor)
    all_indices = np.arange(model.N)
    residuals = (model.data_gradients(theta, all_indices)-model.data_gradients(anchor.theta, all_indices)
                 -model.data_hvps(anchor.theta, theta-anchor.theta, all_indices))
    V = np.mean(np.sum((residuals-residuals.mean(axis=0))**2, axis=1))
    expected = model.N**2/n*(model.N-n)/(model.N-1)*V
    errors = [est.gradient(theta, np.array(batch))-model.full_gradient(theta)
              for batch in combinations(range(model.N), n)]
    observed = np.mean(np.sum(np.asarray(errors)**2, axis=1))
    assert_allclose(observed, expected, atol=1e-24, rtol=1e-10)
    Mi = np.linalg.norm(model.X, axis=1)**3/(6*np.sqrt(3))
    bound = model.N/(4*n)*np.sum(Mi**2)*np.linalg.norm(theta-anchor.theta)**4
    assert observed <= bound+1e-24


def test_replica_and_paired_bootstrap_preserve_pairing_and_failures():
    x = np.arange(40., dtype=float).reshape(20, 2)
    stat = lambda z: z.mean(axis=0)
    single = replica_bootstrap(x, stat, repeats=40, seed=1)
    assert_allclose(single["estimate"], x.mean(axis=0))
    assert np.all(single["lower"] < single["upper"])
    paired = paired_bootstrap(x+3, x, stat, repeats=40, seed=2)
    assert_allclose(paired["estimate"], 3)
    assert_allclose(paired["lower"], 3)
    assert_allclose(paired["upper"], 3)
    cluster = np.stack([x, 2*x, 3*x])
    hierarchical = hierarchical_paired_bootstrap(cluster+2, cluster, lambda s, z: z.mean(axis=0), repeats=40)
    assert_allclose(hierarchical["estimate"], 2)
    assert_allclose(hierarchical["lower"], 2)
    broken = x.copy(); broken[0, 0] = np.nan
    assert replica_bootstrap(broken, stat)["status"] == "failed_ensemble"
    assert paired_bootstrap(broken, x, stat)["status"] == "failed_ensemble"
    cluster[0, 0, 0] = np.nan
    assert hierarchical_paired_bootstrap(cluster, cluster, lambda s, z: z.mean(axis=0))["status"] == "failed_ensemble"


def test_updated_protocol_and_claim_quarantine():
    root = Path(__file__).resolve().parents[1]
    c = json.loads((root/"configs/p0_protocol_v2.json").read_text())
    validate_protocol(c)
    assert c["arms"]["diagonal_cv2"]["method"] == "cv2"
    del c["arms"]["diagonal_cv2"]
    with pytest.raises(ValueError, match="incomplete"):
        validate_protocol(c)
    claims = json.loads((root/"configs/p0_theory_claims.json").read_text())
    assert claims["P1_empirical_comparisons_allowed"]
    assert not claims["publication_theory_certified"]
    assert not claims["theorem_based_superiority_allowed"]


@pytest.mark.parametrize("kind", ["full", "diagonal"])
def test_multistep_gaussian_population_law(kind):
    model = GaussianRegression([[1., .5], [.3, 1.4], [1.2, -.2]], [0., 0., 0.], prior_precision=.7)
    anchor = supplied_anchor([0., 0.], provenance="Gaussian law control")
    est = prepare_estimator(model, "ula", anchor)
    P, cost, _ = fixed_anchor_metric(model, anchor, est, kind=kind)
    options = (dict(preconditioner="frozen_full", frozen_matrix=P) if kind == "full" else
               dict(preconditioner="frozen", frozen_diagonal=P))
    matrix = P if kind == "full" else np.diag(P)
    h, K = .02, 12
    Q = model.full_hessian(anchor.theta)
    A = np.eye(model.d)-h*matrix@Q
    initial = np.array([.2, -.3])
    mean, covariance = initial.copy(), np.zeros((2, 2))
    for _ in range(K):
        mean = A@mean
        covariance = A@covariance@A.T+2*h*matrix
    run = run_with_theory_step(est, h=h, initial=initial, batch_size=2, steps=K,
                              replicas=8192, seed=142950, metric_cost=cost, **options)
    assert_allclose(run.states[-1].mean(axis=0), mean, atol=.02)
    measured = np.cov(run.states[-1], rowvar=False, ddof=0)
    assert np.linalg.norm(measured-covariance)/np.linalg.norm(covariance) < .05

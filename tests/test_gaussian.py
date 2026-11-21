import numpy as np
from numpy.testing import assert_allclose
import pytest

from sgld_cv import GaussianRegression, supplied_anchor, prepare_estimator, run_replicas
from sgld_cv.gaussian import tridiagonal_covariance, make_gaussian_target, analytic_ula_w2
from sgld_cv.transport import empirical_w2


@pytest.mark.parametrize("d,kappa", [(2, 1), (2, 1000), (20, 100), (100, 100000)])
def test_exact_tridiagonal_condition_number(d, kappa):
    covariance, rho = tridiagonal_covariance(d, kappa)
    assert_allclose(np.diag(covariance), 1)
    assert_allclose(np.diag(covariance, 1), rho)
    assert_allclose(np.linalg.cond(covariance), kappa, rtol=1e-8)
    assert np.linalg.eigvalsh(covariance)[0] > 0


def test_quadratic_cv2_exact_at_arbitrary_anchor_but_cv1_has_noise():
    target = make_gaussian_target(N=40, d=3, kappa=20, seed=12)
    model = target.model
    anchor = supplied_anchor([0.2, -0.3, 0.1], provenance="deliberately off-mode")
    theta = np.array([0.5, 0.6, -0.4])
    cv1, cv2 = [prepare_estimator(model, method, anchor) for method in ["cv1", "cv2"]]
    estimates = []
    for i in range(model.N):
        assert_allclose(cv2.gradient(theta, [i]), model.full_gradient(theta), atol=2e-12)
        estimates.append(cv1.gradient(theta, [i]))
    assert np.var(estimates, axis=0).sum() > 1e-3
    assert_allclose(model.full_hessian(theta), target.precision, atol=1e-11)
    assert_allclose(model.full_gradient(np.zeros(3)), 0, atol=1e-12)
    assert_allclose(np.linalg.inv(target.precision), target.covariance, atol=1e-12)


def test_equal_observation_hessians_make_cv1_exact():
    model = GaussianRegression(np.ones((5, 1)), np.arange(5))
    estimator = prepare_estimator(model, "cv1", supplied_anchor([0.2], provenance="test"))
    for i in range(5):
        assert_allclose(estimator.gradient(np.array([0.7]), [i]), model.full_gradient(np.array([0.7])))


def test_gaussian_cv2_and_ula_paired_paths_agree():
    target = make_gaussian_target(N=50, d=4, kappa=30, seed=1)
    anchor = supplied_anchor(np.zeros(4), provenance="exact mode")
    kwargs = dict(initial=anchor.theta, step_size=0.002, batch_size=5,
                  steps=20, replicas=4, seed=300, checkpoints=[0, 1, 10, 20])
    cv2 = run_replicas(prepare_estimator(target.model, "cv2", anchor), **kwargs)
    ula = run_replicas(prepare_estimator(target.model, "ula", anchor), **kwargs)
    assert_allclose(cv2.states, ula.states, atol=1e-13)


def test_analytic_ula_recursion_matches_direct_covariance_recursion():
    target = make_gaussian_target(N=20, d=2, kappa=10, seed=5)
    h, K = 0.03, 50
    values = analytic_ula_w2(target, h, [0, K])
    assert_allclose(values[0], np.sqrt(2), atol=1e-12)
    A = np.eye(2) - h * target.precision / 2
    S = np.zeros((2, 2))
    for _ in range(K):
        S = A @ S @ A.T + h * np.eye(2)
    # S and Sigma commute here, so their eigenvalue pairing gives exact W2.
    e, V = np.linalg.eigh(target.covariance)
    direct = np.linalg.norm(np.sqrt(np.diag(V.T @ S @ V)) - np.sqrt(e))
    assert_allclose(values[-1], direct, atol=1e-12)
    with pytest.raises(ValueError):
        analytic_ula_w2(target, 4.1 / np.linalg.eigvalsh(target.precision)[-1], [1])


def test_pot_w2_has_correct_units_and_known_1d_answer():
    x = np.array([[0.0], [1.0], [2.0]])
    assert_allclose(empirical_w2(x, x), 0, atol=1e-12)
    assert_allclose(empirical_w2(x, x + 3), 3, atol=1e-12)  # W2, not W2 squared
    y = np.array([[-1.0], [0.5], [4.0]])
    assert_allclose(empirical_w2(x, y), np.sqrt(np.mean((x - y) ** 2)), atol=1e-12)
    with pytest.raises(ValueError):
        empirical_w2([[np.nan]], [[0]])


def test_reference_baseline_and_empirical_gaussian_control():
    target = make_gaussian_target(N=32, d=2, kappa=5, seed=4)
    rng = np.random.default_rng(55)
    # Equal target draws must have nonzero empirical OT error.
    a, b = target.sample(512, rng), target.sample(512, rng)
    assert 0 < empirical_w2(a, b) < 0.5
    # From a point mass, the empirical cost is exactly the mean squared radius.
    assert_allclose(empirical_w2(np.zeros_like(a), a), np.sqrt(np.mean(np.sum(a * a, axis=1))), atol=1e-12)
    assert abs(empirical_w2(np.zeros_like(a), a) - np.sqrt(2)) < 0.12


def test_replica_moments_match_exact_finite_step_law():
    target = make_gaussian_target(N=12, d=2, kappa=5, seed=8)
    h, K = .08, 20
    run = run_replicas(prepare_estimator(target.model, "ula"), initial=np.zeros(2),
                       step_size=h, batch_size=4, steps=K, replicas=2048, seed=81)
    A = np.eye(2) - h * target.precision / 2
    covariance = np.zeros((2, 2))
    for _ in range(K):
        covariance = A @ covariance @ A.T + h * np.eye(2)
    assert_allclose(run.states[-1].mean(axis=0), 0, atol=.045)
    assert_allclose(np.cov(run.states[-1], rowvar=False), covariance, atol=.05)

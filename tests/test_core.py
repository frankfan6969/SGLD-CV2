import ast
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.special import expit

from sgld_cv import LogisticRegression, GaussianRegression, supplied_anchor, find_anchor, prepare_estimator
from sgld_cv.costs import CostLedger
from sgld_cv.randomness import generator, BATCH, NOISE
from sgld_cv.samplers import run_replicas, run_chains


@pytest.fixture
def model():
    rng = np.random.default_rng(37)
    return LogisticRegression(rng.normal(size=(7, 3)), np.array([0, 1, 1, 0, 1, 0, 0]), prior_precision=0.7)


def test_logistic_derivatives_and_extreme_logits(model):
    theta = np.array([0.2, -0.4, 0.3])
    step = 1e-5
    basis = np.eye(model.d) * step
    fd_gradient = np.array([(model.potential(theta + e) - model.potential(theta - e)) / (2 * step) for e in basis])
    fd_hessian = np.column_stack([(model.full_gradient(theta + e) - model.full_gradient(theta - e)) / (2 * step) for e in basis])
    assert_allclose(model.full_gradient(theta), fd_gradient, atol=1e-8)
    assert_allclose(model.full_hessian(theta), fd_hessian, atol=1e-8)
    assert np.isfinite(model.potential(theta * 1e6))


@pytest.mark.parametrize("method", ["sgld", "cv1", "cv2"])
def test_exact_batch_enumeration_is_unbiased_and_full_batch_matches(model, method):
    anchor = supplied_anchor([0.1, -0.2, 0.05], provenance="test")
    estimator = prepare_estimator(model, method, anchor)
    theta = np.array([0.3, 0.7, -0.5])
    estimates = [estimator.gradient(theta, np.array(batch)) for batch in combinations(range(model.N), 2)]
    assert_allclose(np.mean(estimates, axis=0), model.full_gradient(theta), atol=1e-12)
    assert_allclose(estimator.gradient(theta, np.arange(model.N)), model.full_gradient(theta), atol=1e-12)


@pytest.mark.parametrize("method", ["cv1", "cv2"])
def test_anchor_variance_is_zero_even_when_anchor_is_not_mode(model, method):
    anchor = supplied_anchor([0.3, 0.4, -0.2], provenance="nonoptimal test anchor")
    estimator = prepare_estimator(model, method, anchor)
    for i in range(model.N):
        assert_allclose(estimator.gradient(anchor.theta, [i]), model.full_gradient(anchor.theta), atol=1e-12)


def test_extracted_estimators_match_original_notebook(model):
    path = Path(__file__).resolve().parents[1] / "STATS 221 Final Project/221_Proj_Frank.ipynb"
    if not path.exists():
        pytest.skip("Historical notebook is not included in the source-only release")
    notebook = json.loads(path.read_text())
    tree = ast.parse("".join(notebook["cells"][5]["source"]))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LogisticRealDataSGLDCVEngine")
    namespace = {"np": np, "expit": expit}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    old = namespace[cls.name].__new__(namespace[cls.name])
    old.X, old.y, old.N, old.d, old.lam = model.X, model.y, model.N, model.d, model.prior_precision
    old.theta_hat = np.array([0.1, -0.2, 0.3])
    old.residual_hat = model.residual(old.theta_hat)
    old.w_hat = model.weights(old.theta_hat)
    old.grad_data_full_hat = model.X.T @ old.residual_hat
    old.hess_data_full_hat = (model.X.T * old.w_hat) @ model.X
    anchor = supplied_anchor(old.theta_hat, provenance="notebook regression check")
    theta, batch = np.array([0.3, 0.2, -0.2]), np.array([0, 2, 6])
    for method in ["sgld", "cv1", "cv2"]:
        assert_allclose(prepare_estimator(model, method, anchor).gradient(theta, batch),
                        old.estimate_gradient(theta, batch, method), atol=1e-12)


def test_dense_hessian_matches_per_observation_hvps(model):
    anchor = supplied_anchor([0.1, 0.2, 0.3], provenance="test")
    estimator = prepare_estimator(model, "cv2", anchor)
    delta = np.array([0.3, -0.2, 0.1])
    assert_allclose(estimator.hessian_hat @ delta,
                    model.data_hvps(anchor.theta, delta, np.arange(model.N)).sum(axis=0), atol=1e-12)


def test_method_specific_caches_and_setup_costs(model):
    cost = CostLedger(seconds={"anchor_optimization": 2.0}, operations={"observation_gradient_evaluations": 35})
    anchor = supplied_anchor(np.zeros(model.d), provenance="measured external optimizer", cost=cost)
    s, c1, c2 = [prepare_estimator(model, method, anchor) for method in ["sgld", "cv1", "cv2"]]
    assert s.cache_bytes == 0
    assert c1.hessian_hat is None and c1.weights_hat is None
    assert c2.hessian_hat.shape == (model.d, model.d)
    assert s.setup_cost.operations["observation_gradient_evaluations"] == 35
    assert c1.setup_cost.operations["observation_gradient_evaluations"] == 35 + model.N
    assert "observation_hessian_evaluations" not in c1.setup_cost.operations
    assert c2.setup_cost.operations["observation_hessian_evaluations"] == model.N
    run = run_replicas(c2, initial=anchor.theta, step_size=1e-3, batch_size=2,
                       steps=5, replicas=3, seed=4, burn_in=2, checkpoints=[0, 2, 5])
    final = run.costs[-1]
    assert final["seconds"]["anchor_optimization"] == 2.0  # once, not once per replica
    assert final["operations"]["observation_gradient_evaluations"] == 35 + model.N + 3 * 5 * 2
    assert final["operations"]["cached_observation_hessian_vector_products"] == 3 * 5 * 2
    assert final["operations"]["dense_hessian_vector_products"] == 3 * 5
    assert final["seconds"]["burn_in"] > 0 and final["seconds"]["sampling"] > 0
    assert_allclose(final["total_seconds"], sum(final["seconds"].values()))
    assert cost.operations["observation_gradient_evaluations"] == 35  # input not mutated
    assert c2.setup_cost.operations["observation_gradient_evaluations"] == 35 + model.N


def test_mode_optimization_is_recorded(model):
    anchor = find_anchor(model)
    assert np.linalg.norm(model.full_gradient(anchor.theta)) < 1e-5
    assert anchor.metadata["success"]
    assert anchor.cost.seconds["anchor_optimization"] > 0
    assert anchor.cost.operations["observation_gradient_evaluations"] >= model.N
    assert anchor.cost.operations["dense_hessian_assemblies"] > 0


def test_shared_randomness_and_independent_replicas(model):
    anchor = supplied_anchor(np.zeros(model.d), provenance="test")
    initial = np.array([0.1, 0.2, -0.3])
    h, seed, b = 0.002, 300, 3
    for method in ["cv2", "sgld", "cv1"]:  # algorithm order does not affect streams
        estimator = prepare_estimator(model, method, anchor)
        run = run_replicas(estimator, initial=initial, step_size=h, batch_size=b,
                           steps=1, replicas=3, seed=seed)
        for r in range(3):
            idx = generator(seed, r, BATCH).choice(model.N, size=b, replace=False)
            z = generator(seed, r, NOISE).standard_normal(model.d)
            assert_allclose(run.states[-1, r], initial - h * estimator.gradient(initial, idx) / 2 + np.sqrt(h) * z)
        repeat = run_replicas(estimator, initial=initial, step_size=h, batch_size=b,
                              steps=1, replicas=3, seed=seed)
        assert np.array_equal(run.states, repeat.states)
        assert not np.array_equal(run.states[-1, 0], run.states[-1, 1])


def test_full_batch_paired_trajectories_and_burnin_slicing(model):
    anchor = supplied_anchor(np.zeros(model.d), provenance="test")
    kwargs = dict(initial=anchor.theta, step_size=0.01, batch_size=model.N, steps=10,
                  replicas=2, seed=7, checkpoints=np.arange(11))
    runs = [run_replicas(prepare_estimator(model, method, anchor), **kwargs) for method in ["sgld", "cv1", "cv2", "ula"]]
    for run in runs[1:]:
        assert_allclose(run.states, runs[0].states, atol=1e-13)
    draws, run = run_chains(prepare_estimator(model, "cv2", anchor), initial=anchor.theta,
                           step_size=0.01, batch_size=model.N, warmup=4, draws=6, chains=2, seed=7)
    assert draws.shape == (2, 6, model.d)
    assert_allclose(draws, runs[2].states[5:].transpose(1, 0, 2))


def test_bad_batch_is_not_silently_capped(model):
    estimator = prepare_estimator(model, "sgld")
    with pytest.raises(ValueError):
        run_replicas(estimator, initial=np.zeros(model.d), step_size=0.01,
                     batch_size=model.N + 1, steps=1, replicas=2, seed=1)


def test_nonfinite_replicas_are_retained_as_failures():
    model = GaussianRegression([[1e100], [1e100]], [0, 0])
    estimator = prepare_estimator(model, "sgld")
    run = run_replicas(estimator, initial=[1e150], step_size=1, batch_size=1,
                       steps=2, replicas=2, seed=1)
    assert len(run.failures) == 2
    assert np.isnan(run.states[-1]).all()


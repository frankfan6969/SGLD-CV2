"""Li et al. 2016 RMSprop, practical Gamma-omitted variant.

The metric uses the RAW mean likelihood gradient, never the CV residual.
Its sign is immaterial after squaring. State is separate for every replica.
"""
from dataclasses import dataclass
import numpy as np
from .costs import CostLedger
from .randomness import generator


@dataclass
class MetricPilot:
    second_moment: np.ndarray
    diagonal: np.ndarray
    cost: CostLedger
    metadata: dict


def fixed_anchor_metric(model, anchor, estimator=None, *, kind="full",
                        relative_ridge=1e-10, absolute_ridge=0.0):
    """P=(H+delta I)^-1 or inverse diagonal, with a frozen prior-scaled ridge.

    delta=max(absolute_ridge, relative_ridge*prior_precision). This rule requires
    no hidden spectral pilot. Invalid curvature fails instead of auto-retuning.
    The ridge changes only the sampler metric, never the CV2 Taylor Hessian or
    posterior target. Diagonal-only CV1 construction is O(Nd), not O(Nd^2).
    """
    if kind not in {"full", "diagonal"}:
        raise ValueError("fixed metric kind must be full or diagonal")
    if (not np.isfinite([relative_ridge, absolute_ridge]).all() or
            relative_ridge < 0 or absolute_ridge < 0):
        raise ValueError("ridge parameters must be finite and nonnegative")
    delta = max(absolute_ridge, relative_ridge * model.prior_precision)
    cost = CostLedger()
    cached = getattr(estimator, "hessian_hat", None)
    if cached is not None:
        with cost.measure("fixed_metric_hessian_reuse"):
            curvature = cached.copy() if kind == "full" else np.diag(cached).copy()
            cost.count("cached_hessian_reuses")
    else:
        with cost.measure("fixed_metric_hessian"):
            weights = model.weights(anchor.theta)
            curvature = ((model.X.T * weights) @ model.X if kind == "full" else
                         (model.X * model.X).T @ weights)
            cost.count("observation_hessian_evaluations", model.N)
            cost.count("dense_hessian_assemblies" if kind == "full" else "diagonal_hessian_assemblies")
            if kind == "diagonal":
                cost.count("observation_diagonal_hessian_evaluations", model.N)
            if model.weights_require_design_product:
                cost.count("batch_design_vector_products")
            cost.count("data_row_visits", model.N)
    with cost.measure("fixed_metric_regularization"):
        if kind == "full":
            curvature = (curvature + curvature.T)/2 + (model.prior_precision + delta)*np.eye(model.d)
            np.linalg.cholesky(curvature)  # validate SPD; separate from noise factorization
            cost.count("cholesky_factorizations")
        else:
            curvature += model.prior_precision + delta
        if not np.isfinite(curvature).all() or (kind == "diagonal" and np.any(curvature <= 0)):
            raise ValueError("fixed curvature must be finite and positive definite")
        cost.count("metric_regularizations")
    with cost.measure("fixed_metric_inverse"):
        if kind == "full":
            metric = np.linalg.solve(curvature, np.eye(model.d))
            metric = (metric + metric.T)/2
            cost.count("dense_matrix_solves")
        else:
            metric = 1/curvature
            cost.count("diagonal_reciprocals", model.d)
    return metric, cost, {"kind": kind, "ridge_shift": delta,
                         "ridge_rule": "max(absolute_ridge, relative_ridge*prior_precision)",
                         "cv2_correction": "full unregularized data Hessian; unchanged"}


def full_hessian_metric(model, anchor, estimator=None):
    """Build P=H(anchor)^-1, reusing CV2's data Hessian when available.

    Anchor optimization remains in the estimator ledger, not this ledger.
    This also works for SGLD/ULA, whose estimator has no theta_hat cache.
    The sampler separately charges the noise Cholesky factorization once.
    """
    cost = CostLedger()
    if estimator is not None and estimator.hessian_hat is not None:
        with cost.measure("full_metric_hessian_reuse"):
            hessian = estimator.hessian_hat + model.prior_precision * np.eye(model.d)
            cost.count("cached_hessian_reuses")
    else:
        with cost.measure("full_metric_hessian"):
            hessian = model.full_hessian(anchor.theta)
            cost.count("observation_hessian_evaluations", model.N)
            cost.count("dense_hessian_assemblies")
            if model.weights_require_design_product:
                cost.count("batch_design_vector_products")
            cost.count("data_row_visits", model.N)
    with cost.measure("full_metric_inverse"):
        matrix = np.linalg.solve(hessian, np.eye(model.d))
        matrix = (matrix + matrix.T) / 2
        cost.count("dense_matrix_solves")
    return matrix, cost


def metric_pilot(model, theta, *, batch_size, batches=100, seed=8100, alpha=.99, epsilon=1e-5):
    """Independent batches at the anchor initialize a common metric.

    This is an explicitly labeled initialization extension to Li's v0=0.
    No posterior reference or evaluation random numbers are used.
    """
    if not 1 <= batch_size <= model.N or batches < 1:
        raise ValueError("invalid metric pilot batch size/count")
    rng, cost = generator(seed, 0, 80), CostLedger()
    metric = RMSprop(model.d, alpha=alpha, epsilon=epsilon)
    with cost.measure("preconditioner_setup"):
        for _ in range(batches):
            idx = rng.choice(model.N, batch_size, replace=False)
            mean = model.X[idx].T @ model.residual(theta, idx) / batch_size
            metric.update(mean)
            cost.count("observation_gradient_evaluations", batch_size)
            cost.count("data_row_visits", batch_size)
            cost.count("metric_updates")
            cost.count("batch_design_vector_products")
            cost.count("metric_batch_gradient_aggregations")
    return MetricPilot(metric.v.copy(), metric.diagonal.copy(), cost,
                       {"seed": seed, "batches": batches, "batch_size": batch_size,
                        "initialization": "RMSprop raw mean likelihood gradients at anchor"})


class RMSprop:
    def __init__(self, d, *, alpha=.99, epsilon=1e-5, initial_v=None):
        if not 0 <= alpha < 1 or not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("require 0 <= alpha < 1 and epsilon > 0")
        self.alpha, self.epsilon = alpha, epsilon
        self.v = np.zeros(d) if initial_v is None else np.array(initial_v, dtype=float, copy=True)
        if self.v.shape != (d,) or not np.isfinite(self.v).all() or np.any(self.v < 0):
            raise ValueError("invalid initial RMSprop second moment")

    @property
    def diagonal(self):
        return 1 / (self.epsilon + np.sqrt(self.v))

    def update(self, raw_mean):
        self.v = self.alpha * self.v + (1 - self.alpha) * np.asarray(raw_mean) ** 2
        return self.diagonal

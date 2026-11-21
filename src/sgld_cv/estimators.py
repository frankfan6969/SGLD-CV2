"""Standalone estimators. CV1 never constructs curvature caches.

Dense CV2 stores one d by d Hessian and O(N) scalar weights, never N Hessians.
"""
from dataclasses import dataclass
import numpy as np
from .costs import CostLedger


@dataclass
class GradientEstimator:
    model: object
    method: str
    setup_cost: CostLedger
    theta_hat: np.ndarray | None = None
    residual_hat: np.ndarray | None = None
    gradient_hat: np.ndarray | None = None
    weights_hat: np.ndarray | None = None
    hessian_hat: np.ndarray | None = None

    @property
    def cache_bytes(self):
        return sum(x.nbytes for x in [self.theta_hat, self.residual_hat, self.gradient_hat,
                                     self.weights_hat, self.hessian_hat] if x is not None)

    def gradient(self, theta, indices, cost: CostLedger | None = None, *, return_raw_mean=False):
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.model.d,):
            raise ValueError("theta has wrong dimension")
        if self.method == "ula":
            if cost is not None:
                cost.count("observation_gradient_evaluations", self.model.N)
                cost.count("prior_gradient_evaluations")
                cost.count("data_row_visits", self.model.N)
                cost.count("batch_design_vector_products")
                cost.count("batch_gradient_aggregations")
            g = self.model.full_gradient(theta)
            return (g, (g - self.model.prior_gradient(theta)) / self.model.N) if return_raw_mean else g
        indices = np.asarray(indices)
        if (indices.ndim != 1 or len(indices) == 0 or indices.dtype.kind not in "iu"
                or np.any(indices < 0) or np.any(indices >= self.model.N)):
            raise ValueError("indices must be nonempty valid integer observation indices")
        Xb = self.model.X[indices]
        residual = self.model.residual(theta, indices)
        b = len(indices)
        raw_mean = Xb.T @ residual / b if return_raw_mean else None
        if return_raw_mean and cost is not None:
            cost.count("metric_batch_gradient_aggregations")
        if cost is not None:
            cost.count("observation_gradient_evaluations", b)
            cost.count("prior_gradient_evaluations")
            cost.count("data_row_visits", b)
            cost.count("batch_design_vector_products")
        if self.method == "sgld":
            data_gradient = (self.model.N * raw_mean if return_raw_mean else
                             (self.model.N / b) * (Xb.T @ residual))
            if cost is not None and not return_raw_mean:
                cost.count("batch_gradient_aggregations")
        else:
            remainder = residual - self.residual_hat[indices]
            data_gradient = self.gradient_hat.copy()
            if self.method == "cv2":
                delta = theta - self.theta_hat
                remainder = remainder - self.weights_hat[indices] * (Xb @ delta)
                data_gradient += self.hessian_hat @ delta
                if cost is not None:
                    cost.count("batch_design_vector_products")
                    cost.count("cached_observation_hessian_vector_products", b)
                    cost.count("dense_hessian_vector_products")
            data_gradient += (self.model.N / b) * (Xb.T @ remainder)
            if cost is not None:
                cost.count("batch_gradient_aggregations")
        gradient = data_gradient + self.model.prior_gradient(theta)
        return (gradient, raw_mean) if return_raw_mean else gradient


def prepare_estimator(model, method, anchor=None, *, charge_anchor=True):
    """Prepare just this method's state, with setup charged once per collection.

    SGLD/ULA also inherit anchor costs when using its location to initialize.
    charge_anchor=False is for explicitly reported conditional-on-anchor runs.
    """
    if method not in {"sgld", "cv1", "cv2", "ula"}:
        raise ValueError("method must be sgld, cv1, cv2, or ula")
    if method in {"cv1", "cv2"} and anchor is None:
        raise ValueError("CV estimators require an anchor")
    if anchor is not None and anchor.theta.shape != (model.d,):
        raise ValueError("anchor dimension does not match model")
    cost = anchor.cost.copy() if anchor is not None and charge_anchor else CostLedger()
    estimator = GradientEstimator(model, method, cost)
    if method in {"cv1", "cv2"}:
        with cost.measure("cv_anchor_gradient"):
            estimator.theta_hat = anchor.theta.copy()
            estimator.residual_hat = model.residual(anchor.theta)
            estimator.gradient_hat = model.X.T @ estimator.residual_hat
            cost.count("observation_gradient_evaluations", model.N)
            cost.count("data_row_visits", model.N)
            cost.count("batch_design_vector_products")
            cost.count("batch_gradient_aggregations")
        if method == "cv2":
            with cost.measure("cv_anchor_hessian"):
                estimator.weights_hat = model.weights(anchor.theta)
                estimator.hessian_hat = (model.X.T * estimator.weights_hat) @ model.X
                cost.count("observation_hessian_evaluations", model.N)
                cost.count("dense_hessian_assemblies")
                cost.count("data_row_visits", model.N)
                if model.weights_require_design_product:
                    cost.count("batch_design_vector_products")
    return estimator

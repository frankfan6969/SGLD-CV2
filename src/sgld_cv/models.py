"""Finite-sum models: U = lambda ||theta||²/2 + sum_i U_i.

No implicit intercept, rescaling by N, optimization, or caches at construction.
The logistic algebra is extracted from notebook cell 5, with a stable objective.
"""
import numpy as np
from scipy.special import expit


class RegressionModel:
    def __init__(self, X, y, prior_precision=1.0):
        self.X = np.array(X, dtype=np.float64, copy=True)
        self.y = np.array(y, dtype=np.float64, copy=True)
        if self.X.ndim != 2 or min(self.X.shape) < 1:
            raise ValueError("X must be a nonempty N by d matrix")
        self.N, self.d = self.X.shape
        if self.y.shape != (self.N,) or not np.isfinite(self.X).all() or not np.isfinite(self.y).all():
            raise ValueError("finite X and one finite response per row are required")
        if not np.isfinite(prior_precision) or prior_precision <= 0:
            raise ValueError("prior_precision must be positive")
        self.prior_precision = float(prior_precision)
        self.X.flags.writeable = self.y.flags.writeable = False

    def prior_gradient(self, theta):
        return self.prior_precision * theta

    def full_gradient(self, theta):
        return self.X.T @ self.residual(theta) + self.prior_gradient(theta)

    def full_hessian(self, theta):
        return (self.X.T * self.weights(theta)) @ self.X + self.prior_precision * np.eye(self.d)

    def data_gradients(self, theta, indices):
        return self.residual(theta, indices)[:, None] * self.X[indices]

    def data_hvps(self, theta, delta, indices):
        Xb = self.X[indices]
        return (self.weights(theta, indices) * (Xb @ delta))[:, None] * Xb


class LogisticRegression(RegressionModel):
    weights_require_design_product = True
    def __init__(self, X, y, prior_precision=1.0):
        super().__init__(X, y, prior_precision)
        if not np.isin(self.y, [0, 1]).all():
            raise ValueError("logistic labels must be 0 or 1")

    def residual(self, theta, indices=None):
        X, y = (self.X, self.y) if indices is None else (self.X[indices], self.y[indices])
        return expit(X @ theta) - y

    def weights(self, theta, indices=None):
        X = self.X if indices is None else self.X[indices]
        p = expit(X @ theta)
        return p * (1 - p)

    def potential(self, theta):
        logits = self.X @ theta
        return float(np.sum(np.logaddexp(0, logits) - self.y * logits)
                     + 0.5 * self.prior_precision * (theta @ theta))


class GaussianRegression(RegressionModel):
    weights_require_design_product = False
    def __init__(self, X, y, prior_precision=1.0, noise_variance=1.0):
        super().__init__(X, y, prior_precision)
        if not np.isfinite(noise_variance) or noise_variance <= 0:
            raise ValueError("noise_variance must be positive and fixed")
        self.noise_variance = float(noise_variance)

    def residual(self, theta, indices=None):
        X, y = (self.X, self.y) if indices is None else (self.X[indices], self.y[indices])
        return (X @ theta - y) / self.noise_variance

    def weights(self, theta, indices=None):
        size = self.N if indices is None else len(indices)
        return np.full(size, 1 / self.noise_variance)

    def potential(self, theta):
        residual = self.X @ theta - self.y
        return float(0.5 * (residual @ residual) / self.noise_variance
                     + 0.5 * self.prior_precision * (theta @ theta))


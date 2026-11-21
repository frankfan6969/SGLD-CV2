"""The instructor's tridiagonal covariance, realized as a quadratic finite sum."""
from dataclasses import dataclass
import numpy as np
from .models import GaussianRegression
from .randomness import generator, DATASET


def tridiagonal_covariance(d, kappa, rho_sign=1):
    if not isinstance(d, (int, np.integer)) or d < 2:
        raise ValueError("d must be an integer >= 2")
    if not np.isfinite(kappa) or kappa < 1 or rho_sign not in (-1, 1):
        raise ValueError("kappa must be finite >= 1; rho_sign must be +/-1")
    rho = rho_sign * ((kappa - 1) / (kappa + 1)) / (2 * np.cos(np.pi / (d + 1)))
    covariance = np.eye(d) + np.diag(np.full(d - 1, rho), 1) + np.diag(np.full(d - 1, rho), -1)
    eigenvalues = np.linalg.eigvalsh(covariance)
    if eigenvalues[0] <= 0:
        raise ValueError("requested condition number is numerically singular")
    if not np.isclose(eigenvalues[-1] / eigenvalues[0], kappa, rtol=1e-7):
        raise ValueError("requested condition number cannot be represented accurately")
    return covariance, float(rho)


@dataclass
class GaussianTarget:
    model: GaussianRegression
    covariance: np.ndarray
    precision: np.ndarray
    mean: np.ndarray
    rho: float
    metadata: dict

    def sample(self, size, rng):
        return self.mean + rng.standard_normal((size, len(self.mean))) @ np.linalg.cholesky(self.covariance).T


def make_gaussian_target(*, N, d, kappa, seed, rho_sign=1, residual_scale=1.0):
    """QR gives X.T X = Q-lambda I; y is projected into null(X.T).

    Varying rank-one observation Hessians preserve CV1 subsampling noise.
    The data-generation cost is experiment construction, not sampler setup.
    """
    if not isinstance(N, int) or N <= d:
        raise ValueError("N must exceed d to permit nonzero observation residuals")
    if not np.isfinite(residual_scale) or residual_scale <= 0:
        raise ValueError("residual_scale must be positive")
    covariance, rho = tridiagonal_covariance(d, kappa, rho_sign)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    precision = (eigenvectors * (1 / eigenvalues)) @ eigenvectors.T
    lam = 0.1 / eigenvalues[-1]
    rng = generator(seed, 0, DATASET)
    orthogonal, _ = np.linalg.qr(rng.standard_normal((N, d)), mode="reduced")
    factor = np.linalg.cholesky(precision - lam * np.eye(d))
    X = orthogonal @ factor.T
    y = rng.standard_normal(N)
    y -= orthogonal @ (orthogonal.T @ y)
    y *= residual_scale * np.sqrt(N) / np.linalg.norm(y)
    model = GaussianRegression(X, y, prior_precision=lam)
    actual_precision = model.full_hessian(np.zeros(d))
    linear_term = X.T @ y
    if not np.allclose(actual_precision, precision, rtol=1e-9, atol=1e-9):
        raise ArithmeticError("finite-sum precision does not match target")
    actual_mean = np.linalg.solve(actual_precision, linear_term)
    if np.linalg.norm(actual_mean) > 1e-8:
        raise ArithmeticError("finite-sum mean differs from zero beyond tolerance")
    return GaussianTarget(model, covariance, precision, np.zeros(d), rho, {
        "N": N, "d": d, "requested_kappa": float(kappa),
        "actual_kappa": float(eigenvalues[-1] / eigenvalues[0]), "rho": rho,
        "prior_precision": lam, "residual_scale": residual_scale, "dataset_seed": seed,
        "precision_relative_error": float(np.linalg.norm(actual_precision - precision) / np.linalg.norm(precision)),
        "finite_sum_mean_error": float(np.linalg.norm(actual_mean)),
        "observation_hessians": "heterogeneous rank-one; exact quadratic CV2 residual",
        "target": "N(0, tridiagonal covariance with diagonal 1)",
    })


def analytic_ula_w2(target, step_size, checkpoints, initial=None):
    """Exact finite-k Gaussian W2 for identity-metric full-gradient ULA only.

    CV2 matches this law here; general SGLD/CV1 laws are not assumed Gaussian.
    """
    eig, V = np.linalg.eigh(target.precision)
    if not 0 < step_size * eig[-1] < 4:
        raise ValueError("ULA requires 0 < h*lambda_max(Q) < 4")
    a = 1 - 0.5 * step_size * eig
    initial = target.mean if initial is None else np.asarray(initial, dtype=float)
    centered = V.T @ (initial - target.mean)
    result = []
    for k in checkpoints:
        # Sum of the geometric series of injected Gaussian noise covariances.
        variance = step_size * (1 - a ** (2 * int(k))) / (1 - a * a)
        mean_squared = np.sum((a ** int(k) * centered) ** 2)
        result.append(np.sqrt(mean_squared + np.sum((np.sqrt(np.maximum(variance, 0)) - 1 / np.sqrt(eig)) ** 2)))
    return np.asarray(result)


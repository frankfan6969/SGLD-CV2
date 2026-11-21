"""Frozen posterior functionals; trajectory averages and replica RMSE.

Second moments are additive functionals. Posterior variances are derived as
E[q^2]-E[q]^2, not mislabeled second moments or Bessel-corrected iid estimates.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import expit


@dataclass
class FunctionalSpec:
    center: np.ndarray
    transform: np.ndarray
    slow_direction: np.ndarray
    prediction_rows: np.ndarray

    @classmethod
    def from_geometry(cls, center, hessian, prediction_rows=None):
        center, hessian = np.asarray(center), np.asarray(hessian)
        eigenvalues, vectors = np.linalg.eigh(hessian)
        if (hessian.shape != (len(center), len(center)) or
                not np.isfinite(hessian).all() or eigenvalues[0] <= 0):
            raise ValueError("functional geometry must be SPD")
        transform = (vectors * np.sqrt(eigenvalues)) @ vectors.T
        rows = np.empty((0, len(center))) if prediction_rows is None else np.asarray(prediction_rows)
        return cls(center.copy(), transform, vectors[:, 0] * np.sqrt(eigenvalues[0]), rows.copy())

    @property
    def names(self):
        d = len(self.center)
        return ([f"coordinate_mean_{j}" for j in range(d)] +
                [f"coordinate_second_moment_{j}" for j in range(d)] +
                ["slow_mean", "slow_second_moment"] +
                [f"prediction_{j}" for j in range(len(self.prediction_rows))])

    def __call__(self, theta):
        theta = np.asarray(theta, dtype=float)
        q = (theta - self.center) @ self.transform
        slow = (theta - self.center) @ self.slow_direction
        return np.concatenate([q, q*q, slow[..., None], (slow*slow)[..., None],
                               expit(theta @ self.prediction_rows.T)], axis=-1)

    def derive(self, averages):
        values = np.array(averages, copy=True)
        d = len(self.center)
        values[..., d:2*d] -= values[..., :d] ** 2
        values[..., 2*d+1] -= values[..., 2*d] ** 2
        names = [n.replace("second_moment", "variance") for n in self.names]
        return values, names


def functional_rmse(averages, target, scales, names):
    """No failed replica is removed. Scales are frozen before sampler runs."""
    averages, target, scales = map(np.asarray, (averages, target, scales))
    if (averages.ndim != 2 or averages.shape[1:] != target.shape or scales.shape != target.shape
            or len(names) != len(target) or not np.isfinite(target).all()
            or not np.isfinite(scales).all() or np.any(scales <= 0)):
        raise ValueError("invalid functional RMSE dimensions/scales")
    if not np.isfinite(averages).all():
        return [{"functional": name, "status": "failed_ensemble", "rmse": None,
                 "normalized_rmse": None} for name in names]
    errors = averages - target
    return [{"functional": name, "status": "ok", "rmse": float(np.sqrt(np.mean(errors[:, j]**2))),
             "normalized_rmse": float(np.sqrt(np.mean(errors[:, j]**2)) / scales[j]),
             "bias": float(errors[:, j].mean()),
             "replica_sd": float(averages[:, j].std(ddof=1)) if len(averages) > 1 else None}
            for j, name in enumerate(names)]

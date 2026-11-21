"""Exact empirical W2 via POT; no Gaussian fit or regularized substitution."""
import warnings
from dataclasses import dataclass
import hashlib
import numpy as np
from scipy.spatial.distance import cdist


def empirical_w2(x, y, *, max_pairs=4_000_000):
    import ot
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if (x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]
            or min(len(x), len(y)) == 0 or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError("W2 requires finite nonempty sample matrices with matching dimensions")
    if len(x) * len(y) > max_pairs:
        raise ValueError("exact OT pair budget exceeded; explicitly increase max_pairs or reduce sample counts")
    C = cdist(x, y, metric="sqeuclidean")
    if not np.isfinite(C).all():
        raise FloatingPointError("squared transport costs overflowed")
    scale = float(C.max())
    if scale == 0:
        return 0.0
    # Scaling improves the LP conditioning. Restore scale before taking sqrt.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        objective, log = ot.emd2(np.full(len(x), 1 / len(x)), np.full(len(y), 1 / len(y)),
                                 C / scale, log=True, numItermax=1_000_000, numThreads=1)
    if log.get("warning") or log.get("result_code", 1) != 1 or not np.isfinite(objective):
        raise RuntimeError(f"POT solver failed: {log}")
    return float(np.sqrt(max(float(objective), 0) * scale))


@dataclass(frozen=True)
class AccuracyMetric:
    """A fixed posterior accuracy metric shared by all methods/checkpoints.

    For sliced W2, calibration must be independent of the samples scored later.
    ``transform`` is the symmetric inverse square root of the calibration
    covariance. Projection columns are fixed unit vectors in whitened space.
    """
    kind: str
    center: np.ndarray | None = None
    transform: np.ndarray | None = None
    projections: np.ndarray | None = None
    covariance_floor: float | None = None

    def distance(self, x, y):
        if self.kind == "w2":
            return empirical_w2(x, y)
        if self.kind != "standardized_sliced_w2":
            raise ValueError(f"unknown accuracy metric {self.kind}")
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        d = len(self.center)
        if (x.ndim != 2 or y.ndim != 2 or x.shape[1] != d or y.shape[1] != d
                or not len(x) or not len(y) or not np.isfinite(x).all() or not np.isfinite(y).all()):
            raise ValueError("sliced W2 requires finite nonempty sample matrices of calibration dimension")
        # Equal-size clouds are typical, but unequal sizes use exact 1D quantile
        # integration rather than interpolation at arbitrary grid points.
        xp = np.sort(((x - self.center) @ self.transform) @ self.projections, axis=0)
        yp = np.sort(((y - self.center) @ self.transform) @ self.projections, axis=0)
        if len(xp) == len(yp):
            squared = np.mean((xp - yp) ** 2, axis=0)
        else:
            knots = np.unique(np.r_[np.arange(len(xp) + 1) / len(xp),
                                     np.arange(len(yp) + 1) / len(yp)])
            mid = (knots[:-1] + knots[1:]) / 2
            xi = np.minimum((mid * len(xp)).astype(int), len(xp) - 1)
            yi = np.minimum((mid * len(yp)).astype(int), len(yp) - 1)
            squared = np.sum((knots[1:] - knots[:-1])[:, None] *
                             (xp[xi] - yp[yi]) ** 2, axis=0)
        return float(np.sqrt(np.mean(squared)))


def make_accuracy_metric(d, *, kind="auto", cutoff=10, center=None,
                         covariance=None, projections=512, seed=926,
                         eigenvalue_floor=1e-8):
    """Use empirical W2 in low dimension, standardized sliced W2 otherwise."""
    if kind == "auto":
        kind = "w2" if d <= cutoff else "standardized_sliced_w2"
    if kind == "w2":
        return AccuracyMetric(kind)
    if kind != "standardized_sliced_w2":
        raise ValueError(f"unknown accuracy metric {kind}")
    center = np.asarray(center, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    if (center.shape != (d,) or covariance.shape != (d, d) or
            not np.isfinite(center).all() or not np.isfinite(covariance).all()):
        raise ValueError("calibration requires a finite mean and d-by-d covariance")
    if not isinstance(projections, int) or projections < 1:
        raise ValueError("projections must be positive")
    if not 0 < eigenvalue_floor < 1:
        raise ValueError("eigenvalue_floor must lie between 0 and 1")
    eigenvalues, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
    floor = float(eigenvalue_floor * max(float(eigenvalues[-1]), 1e-300))
    if eigenvalues[-1] <= 0:
        raise ValueError("calibration covariance must have positive eigenvalues")
    inverse_sqrt = (vectors * (1 / np.sqrt(np.maximum(eigenvalues, floor)))) @ vectors.T
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(d, projections))
    directions /= np.linalg.norm(directions, axis=0)
    return AccuracyMetric(kind, center, inverse_sqrt, directions, floor)


def metric_fingerprint(metric):
    """Stable content hash for a saved whitening/projection calibration."""
    if metric.kind == "w2":
        return None
    digest = hashlib.sha256()
    for array in (metric.center, metric.transform, metric.projections):
        value = np.ascontiguousarray(array, dtype=np.float64)
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def load_accuracy_metric(path, *, d, covariance_floor=None):
    """Load the exact calibration used for pilot tuning."""
    with np.load(path) as saved:
        center = np.asarray(saved["center"], dtype=float)
        transform = np.asarray(saved["transform"], dtype=float)
        projections = np.asarray(saved["projections"], dtype=float)
    if (center.shape != (d,) or transform.shape != (d,d) or projections.ndim != 2 or
            projections.shape[0] != d or not projections.shape[1] or
            not all(np.isfinite(a).all() for a in (center,transform,projections)) or
            not np.allclose(np.linalg.norm(projections,axis=0),1,atol=1e-10)):
        raise ValueError("invalid frozen accuracy calibration")
    return AccuracyMetric("standardized_sliced_w2",center,transform,projections,covariance_floor)

"""Anchor computation with measured cost and optimization provenance."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from .costs import CostLedger


@dataclass
class Anchor:
    theta: np.ndarray
    cost: CostLedger
    metadata: dict


def supplied_anchor(theta, *, provenance: str, cost: CostLedger | None = None):
    """Caller must identify externally supplied truth/initialization and any cost.

    Zero cost is appropriate for a known benchmark mode, not a hidden optimizer.
    """
    theta = np.array(theta, dtype=float, copy=True)
    if theta.ndim != 1 or not np.isfinite(theta).all() or not provenance:
        raise ValueError("finite vector and nonempty provenance required")
    theta.flags.writeable = False
    return Anchor(theta, cost.copy() if cost is not None else CostLedger(),
                  {"source": provenance, "optimized": False})


def find_anchor(model, initial=None, tolerance=1e-8, max_iterations=1000):
    """Newton-CG as in the notebook; count every objective/gradient/Hessian call."""
    cost = CostLedger()
    initial = np.zeros(model.d) if initial is None else np.asarray(initial, dtype=float)
    if initial.shape != (model.d,) or not np.isfinite(initial).all():
        raise ValueError("initial must be a finite d-vector")
    if tolerance <= 0 or max_iterations < 1:
        raise ValueError("invalid optimizer tolerance/iteration limit")

    def objective(theta):
        cost.count("observation_likelihood_evaluations", model.N)
        cost.count("data_row_visits", model.N)
        cost.count("batch_design_vector_products")
        return model.potential(theta)

    def gradient(theta):
        cost.count("observation_gradient_evaluations", model.N)
        cost.count("prior_gradient_evaluations")
        cost.count("data_row_visits", model.N)
        cost.count("batch_design_vector_products")
        cost.count("batch_gradient_aggregations")
        return model.full_gradient(theta)

    def hessian(theta):
        cost.count("observation_hessian_evaluations", model.N)
        cost.count("dense_hessian_assemblies")
        cost.count("data_row_visits", model.N)
        if model.weights_require_design_product:
            cost.count("batch_design_vector_products")
        return model.full_hessian(theta)

    with cost.measure("anchor_optimization"):
        result = minimize(objective, initial, jac=gradient, hess=hessian,
                          method="Newton-CG", options={"xtol": tolerance, "maxiter": max_iterations})
        final_norm = float(np.linalg.norm(gradient(result.x)))
    if not result.success or not np.isfinite(final_norm):
        raise RuntimeError(f"Anchor optimization failed: {result.message}; gradient norm={final_norm}")
    theta = np.array(result.x, copy=True)
    theta.flags.writeable = False
    return Anchor(theta, cost, {"source": "Newton-CG", "optimized": True,
                  "success": bool(result.success), "message": str(result.message),
                  "iterations": int(result.nit), "gradient_norm": final_norm,
                  "tolerance": tolerance})


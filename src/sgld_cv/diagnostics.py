"""Post-burn-in multichain diagnostics and setup-inclusive ESS denominators."""
import numpy as np


def chain_diagnostics(samples, *, cost=None, adaptive=False, rhat_limit=1.01, ess_min=400):
    import arviz as az
    samples = np.asarray(samples)
    if samples.ndim != 3 or samples.shape[0] < 2 or samples.shape[1] < 4:
        raise ValueError("require (chains>=2, retained draws>=4, coordinates)")
    if not np.isfinite(samples).all():
        return {"reliable": False, "reason": "nonfinite retained draws", "coordinates": []}
    rows = []
    C, T, d = samples.shape
    for j in range(d):
        x = samples[:, :, j]
        bulk = float(az.ess(x, method="bulk"))
        tail = float(az.ess(x, method="tail"))
        rhat = float(az.rhat(x, method="rank"))
        mcse = float(np.asarray(az.mcse(x, method="mean")).item())
        finite = np.isfinite([bulk, tail, rhat, mcse]).all()
        row = {"coordinate": j, "ess_bulk": bulk if finite else None,
               "ess_tail": tail if finite else None, "rhat": rhat if finite else None,
               "mcse_mean": mcse if finite else None,
               "ess_per_retained_iteration": bulk / (C * T) if finite else None}
        if cost is not None:
            grads = cost["operations"].get("observation_gradient_evaluations", 0)
            production = cost["seconds"].get("sampling", 0)
            row.update({"ess_per_total_second": bulk / cost["total_seconds"] if finite and cost["total_seconds"] > 0 else None,
                        "ess_per_production_second": bulk / production if finite and production > 0 else None,
                        "ess_per_observation_gradient": bulk / grads if finite and grads else None})
            transitions = cost["operations"].get("transitions", 0)
            row["ess_per_total_transition"] = bulk / transitions if finite and transitions else None
        rows.append(row)
    reliable = all(r["rhat"] is not None and r["rhat"] < rhat_limit and
                   min(r["ess_bulk"], r["ess_tail"]) >= ess_min for r in rows) and not adaptive
    return {"reliable": reliable, "adaptive_exploratory_only": adaptive,
            "reason": "passed diagnostic thresholds (not a posterior accuracy guarantee)" if reliable else
                      "adaptive metric or insufficient mixing/ESS; do not rank as validated throughput",
            "max_rhat": max((r["rhat"] for r in rows if r["rhat"] is not None), default=None),
            "min_bulk_ess": min((r["ess_bulk"] for r in rows if r["ess_bulk"] is not None), default=None),
            "min_tail_ess": min((r["ess_tail"] for r in rows if r["ess_tail"] is not None), default=None),
            "coordinates": rows, "chains": C, "retained_draws_per_chain": T}


def gradient_audit(estimators, states, *, batch_size, batches=100, seed=9000, metric_matrix=None):
    """Shared states/batches; arithmetic MSE, conditional bias and MC uncertainty.

    State-level means are the units for the reported aggregate standard error.
    The raw bias-norm estimate is upward biased at finite batch replication.
    """
    from .randomness import generator
    if not estimators or batches < 2 or len(states) < 1:
        raise ValueError("nonempty estimators/states and at least two batches required")
    model = next(iter(estimators.values())).model
    if not 1 <= batch_size <= model.N:
        raise ValueError("invalid batch size")
    rng = generator(seed, 0, 90)
    if metric_matrix is not None:
        metric_matrix = np.asarray(metric_matrix, dtype=float)
        if (metric_matrix.shape != (model.d, model.d) or not np.isfinite(metric_matrix).all() or
                not np.allclose(metric_matrix, metric_matrix.T)):
            raise ValueError("gradient error metric must be finite symmetric SPD")
        np.linalg.cholesky(metric_matrix)
    errors = {name: np.empty((len(states), batches)) for name in estimators}
    metric_errors = {name: np.empty((len(states), batches)) for name in estimators}
    bias = {name: [] for name in estimators}
    for s, theta in enumerate(states):
        exact = model.full_gradient(theta)
        sums = {name: np.zeros(model.d) for name in estimators}
        for b in range(batches):
            indices = rng.choice(model.N, batch_size, replace=False)
            for name, estimator in estimators.items():
                delta = estimator.gradient(theta, indices) - exact
                errors[name][s, b] = delta @ delta
                metric_errors[name][s, b] = delta @ delta if metric_matrix is None else delta @ metric_matrix @ delta
                sums[name] += delta
        for name in estimators:
            bias[name].append(float(np.linalg.norm(sums[name] / batches)))
    summary = []
    for name, values in errors.items():
        state_means = values.mean(axis=1)
        summary.append({"method": name, "mean_mse": float(values.mean()),
            "median_squared_error": float(np.median(values)), "q95_squared_error": float(np.quantile(values, .95)),
            "max_squared_error": float(values.max()),
            "mse_state_mean_se": float(state_means.std(ddof=1) / np.sqrt(len(states))) if len(states) > 1 else None,
            "mean_empirical_bias_norm": float(np.mean(bias[name])), "states": len(states),
            "mean_preconditioned_mse": float(metric_errors[name].mean()),
            "median_preconditioned_squared_error": float(np.median(metric_errors[name])),
            "q95_preconditioned_squared_error": float(np.quantile(metric_errors[name], .95)),
            "metric_error_definition": "e.T @ P @ e" if metric_matrix is not None else "e.T @ e",
            "batches_per_state": batches, "batch_size": batch_size})
    return summary, errors


def radius_audit(estimators, anchor, *, radii, directions, batch_size, batches, seed):
    directions = np.asarray(directions, dtype=float)
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    rows = []
    for radius in radii:
        summary, _ = gradient_audit(estimators, anchor + radius * directions,
                                   batch_size=batch_size, batches=batches, seed=seed)
        rows.extend(dict(row, radius=float(radius)) for row in summary)
    return rows

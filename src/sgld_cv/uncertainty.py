"""Paired, path-level bootstrap conditional on frozen references and geometry.

Approximate pointwise percentile intervals describe finite-cloud estimators;
they are not guaranteed population-W2 coverage or simultaneous frontier bands.
Never resample time draws as independent replicas. HMC uncertainty is reported
separately and cannot be absorbed by iid resampling of correlated reference rows.
"""
import numpy as np


def replica_bootstrap(samples, statistic, *, repeats=999, seed=142902, level=.95):
    samples = np.asarray(samples)
    if samples.ndim < 2 or len(samples) < 2 or repeats < 20 or not 0 < level < 1:
        raise ValueError("bootstrap needs >=2 independent replicas, >=20 repeats and valid level")
    if not np.isfinite(samples).all():
        return {"status": "failed_ensemble", "estimate": None, "lower": None, "upper": None}
    rng = np.random.default_rng(seed)
    estimate = np.asarray(statistic(samples))
    draws = np.asarray([statistic(samples[rng.integers(0, len(samples), len(samples))]) for _ in range(repeats)])
    if not np.isfinite(draws).all() or not np.isfinite(estimate).all():
        raise ValueError("nonfinite bootstrap statistic")
    lower, upper = np.quantile(draws, [(1-level)/2, (1+level)/2], axis=0)
    return {"status": "ok", "estimate": estimate, "lower": lower, "upper": upper,
            "repeats": repeats, "level": level, "scope": "whole-replica pointwise percentile, conditional on reference"}


def paired_bootstrap(samples_a, samples_b, statistic, *, repeats=999, seed=142900, level=.95):
    a, b = np.asarray(samples_a), np.asarray(samples_b)
    if (a.shape != b.shape or a.ndim < 2 or len(a) < 2 or repeats < 20 or
            not 0 < level < 1):
        raise ValueError("paired bootstrap needs matching replicas, >=20 repeats and valid level")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return {"status": "failed_ensemble", "estimate": None, "lower": None, "upper": None}
    rng = np.random.default_rng(seed)
    observed = np.asarray(statistic(a)) - np.asarray(statistic(b))
    values = []
    for _ in range(repeats):
        indices = rng.integers(0, len(a), size=len(a))
        values.append(np.asarray(statistic(a[indices]))-np.asarray(statistic(b[indices])))
    values = np.asarray(values)
    if not np.isfinite(values).all() or not np.isfinite(observed).all():
        raise ValueError("nonfinite bootstrap statistic")
    low, high = np.quantile(values, [(1-level)/2, (1+level)/2], axis=0)
    return {"status": "ok", "estimate": observed, "lower": low, "upper": high,
            "repeats": repeats, "level": level, "pairing": "same whole-replica indices in both methods",
            "scope": "pointwise percentile, conditional on frozen reference/geometry"}


def hierarchical_paired_bootstrap(a, b, statistic, *, repeats=999, seed=142901, level=.95):
    """Outer algorithm-seed clusters, inner paired independent replica paths.

    statistic takes (seed_index, replica_array), allowing a fixed per-seed OT
    reference. Returns a mean paired contrast across seeds. Use adequate outer
    seeds in production; two-seed smoke intervals are pipeline checks only.
    """
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.ndim < 3 or a.shape[0] < 2 or a.shape[1] < 2 or repeats < 20 or not 0 < level < 1:
        raise ValueError("hierarchical bootstrap requires >=2 seeds/replicas and >=20 repeats")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return {"status": "failed_ensemble", "estimate": None, "lower": None, "upper": None}
    S, R = a.shape[:2]
    rng = np.random.default_rng(seed)
    estimate = np.mean([np.asarray(statistic(s, a[s]))-np.asarray(statistic(s, b[s])) for s in range(S)], axis=0)
    draws = []
    for _ in range(repeats):
        contrasts = []
        for s in rng.integers(0, S, S):
            indices = rng.integers(0, R, R)
            contrasts.append(np.asarray(statistic(s, a[s, indices]))-np.asarray(statistic(s, b[s, indices])))
        draws.append(np.mean(contrasts, axis=0))
    if not np.isfinite(draws).all():
        raise ValueError("nonfinite hierarchical bootstrap statistic")
    lower, upper = np.quantile(draws, [(1-level)/2, (1+level)/2], axis=0)
    return {"status": "ok", "estimate": estimate, "lower": lower, "upper": upper,
            "repeats": repeats, "level": level, "outer_seed_clusters": S,
            "scope": "hierarchical paired pointwise interval; conditional on fixed data/reference"}


def cost_views(snapshot, *, method_tuning_seconds=0.0):
    """Selected-config first-use cost and explicitly separate search-inclusive cost.

    Timing components are nested subsets of sampling and must not be added again.
    """
    if not np.isfinite(method_tuning_seconds) or method_tuning_seconds < 0:
        raise ValueError("tuning seconds must be finite and nonnegative")
    return {"selected_configuration_total_seconds": snapshot["total_seconds"],
            "tuning_seconds": float(method_tuning_seconds),
            "search_inclusive_total_seconds": snapshot["total_seconds"]+method_tuning_seconds,
            "setup_seconds": sum(v for k, v in snapshot["seconds"].items() if k not in {"burn_in", "sampling"}),
            "sampling_seconds": snapshot["seconds"].get("sampling", 0.0),
            "component_seconds_nested_not_additive": snapshot.get("component_seconds", {})}

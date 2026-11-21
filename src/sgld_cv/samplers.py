"""Shared unpreconditioned transition and independent-replica/chain protocols.

theta_next = theta - h*g/2 + sqrt(h)*z. No clipping or hidden stabilization.
"""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from .randomness import generator, BATCH, NOISE
from .preconditioners import RMSprop


@dataclass
class ReplicaRun:
    checkpoints: np.ndarray
    states: np.ndarray  # checkpoint, independent replica, coordinate
    costs: list[dict]
    failures: list[dict]
    metadata: dict
    functional_averages: np.ndarray | None = None  # checkpoint, replica, functional
    adaptive_final_v: np.ndarray | None = None


def run_replicas(estimator, *, initial, step_size, batch_size, steps,
                 replicas, seed, checkpoints=None, burn_in=0, replace=False,
                 preconditioner="identity", alpha=.99, epsilon=1e-5,
                 initial_v=None, frozen_diagonal=None, frozen_matrix=None,
                 metric_cost=None,
                 functionals=None, functional_burn_in=0,
                 failure_transform=None, failure_center=None, max_standardized_radius=None,
                 profile_components=False, record_adaptive_state=False,
                 cost_snapshots=True):
    if not np.isfinite(step_size) or step_size <= 0:
        raise ValueError("step_size must be finite and positive")
    for name, value in [("batch_size", batch_size), ("steps", steps),
                        ("replicas", replicas), ("burn_in", burn_in), ("seed", seed)]:
        if not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if replicas < 1 or batch_size < 1 or burn_in > steps:
        raise ValueError("invalid replica count, batch size, or burn-in")
    if not isinstance(functional_burn_in, int) or not 0 <= functional_burn_in <= steps:
        raise ValueError("functional_burn_in must be an integer within the run")
    if not replace and batch_size > estimator.model.N:
        raise ValueError("batch_size exceeds N without replacement")
    if checkpoints is None:
        checkpoints = [0, steps]
    checkpoints = np.asarray(checkpoints)
    if checkpoints.ndim != 1 or checkpoints.dtype.kind not in "iu":
        raise ValueError("checkpoints must be integer iterations")
    checkpoints = np.unique(np.r_[0, burn_in, checkpoints, steps])
    if np.any(checkpoints < 0) or np.any(checkpoints > steps):
        raise ValueError("checkpoint outside run")
    initial = np.asarray(initial, dtype=float)
    if initial.shape == (estimator.model.d,):
        theta = np.tile(initial, (replicas, 1))
    elif initial.shape == (replicas, estimator.model.d):
        theta = initial.copy()
    else:
        raise ValueError("initial must have shape (d,) or (replicas,d)")
    if not np.isfinite(theta).all():
        raise ValueError("initial states must be finite")
    if max_standardized_radius is not None:
        failure_transform = np.asarray(failure_transform, dtype=float)
        failure_center = np.asarray(failure_center, dtype=float)
        if (failure_transform.shape != (estimator.model.d, estimator.model.d) or
                failure_center.shape != (estimator.model.d,) or
                not np.isfinite(failure_transform).all() or not np.isfinite(failure_center).all() or
                not np.isfinite(max_standardized_radius) or max_standardized_radius <= 0):
            raise ValueError("invalid frozen radius failure rule")
    cost = estimator.setup_cost.copy()
    if metric_cost is not None:
        cost.add(metric_cost)
    if preconditioner not in {"identity", "rmsprop", "frozen", "frozen_full"}:
        raise ValueError("unknown preconditioner")
    if preconditioner == "frozen":
        frozen_diagonal = np.asarray(frozen_diagonal, dtype=float)
        if (frozen_diagonal.shape != (estimator.model.d,) or not np.isfinite(frozen_diagonal).all()
                or np.any(frozen_diagonal <= 0)):
            raise ValueError("frozen metric must be a finite positive d-vector")
    full_cholesky = None
    if preconditioner == "frozen_full":
        frozen_matrix = np.asarray(frozen_matrix, dtype=float)
        if (frozen_matrix.shape != (estimator.model.d, estimator.model.d) or
                not np.isfinite(frozen_matrix).all() or
                not np.allclose(frozen_matrix, frozen_matrix.T, rtol=0, atol=1e-12)):
            raise ValueError("full frozen metric must be a finite symmetric d-by-d matrix")
        with cost.measure("full_metric_factorization"):
            try:
                full_cholesky = np.linalg.cholesky(frozen_matrix)
                cost.count("cholesky_factorizations")
            except np.linalg.LinAlgError as error:
                raise ValueError("full frozen metric must be positive definite") from error
    metrics = None
    if preconditioner == "rmsprop":
        with cost.measure("adaptive_state_initialization"):
            metrics = [RMSprop(estimator.model.d, alpha=alpha, epsilon=epsilon, initial_v=initial_v)
                       for _ in range(replicas)]
            cost.count("adaptive_state_initializations", replicas)
            cost.count("adaptive_state_scalars", replicas*estimator.model.d)
    start = perf_counter()
    components = {"gradient": 0.0, "adaptive_update": 0.0, "drift_and_noise": 0.0}
    batch_rng = [generator(seed, r, BATCH) for r in range(replicas)]
    noise_rng = [generator(seed, r, NOISE) for r in range(replicas)]
    states = np.empty((len(checkpoints), replicas, estimator.model.d))
    states[0] = theta
    live = np.ones(replicas, dtype=bool)
    failures = []
    snapshots = []
    next_checkpoint = 1
    burn_seconds = None
    setup_operations = dict(cost.operations)
    burn_operations = dict(setup_operations)
    functional_seconds = 0.0
    functional_sums = None
    functional_count = 0
    functional_averages = []

    def snapshot(k):
        elapsed = perf_counter() - start - functional_seconds
        if burn_in and k <= burn_in:
            warm, production = elapsed, 0.0
        elif burn_in:
            warm, production = burn_seconds, elapsed - burn_seconds
        else:
            warm, production = 0.0, elapsed
        result = cost.snapshot()
        result["seconds"]["burn_in"] = warm
        result["seconds"]["sampling"] = production
        result["total_seconds"] = sum(result["seconds"].values())
        result["iteration"] = int(k)
        if profile_components:
            result["component_seconds"] = dict(components)  # subset of sampling, never summed into total
        def difference(end, beginning):
            return {name: value - beginning.get(name, 0) for name, value in end.items()
                    if value != beginning.get(name, 0)}
        warm_end = cost.operations if burn_in and k <= burn_in else burn_operations
        result["phase_operations"]["burn_in"] = difference(warm_end, setup_operations)
        result["phase_operations"]["sampling"] = difference(cost.operations, warm_end)
        return result

    snapshots.append(snapshot(0))
    if functionals is not None:
        scoring_start = perf_counter()
        functional_sums = np.zeros_like(functionals(theta))
        functional_averages.append(np.full_like(functional_sums, np.nan))
        functional_seconds += perf_counter() - scoring_start
    for k in range(1, steps + 1):
        for r in range(replicas):
            if not live[r]:
                continue
            # ULA needs no batch, but has the same independent noise stream.
            idx = None if estimator.method == "ula" else batch_rng[r].choice(
                estimator.model.N, size=batch_size, replace=replace)
            z = noise_rng[r].standard_normal(estimator.model.d)
            with np.errstate(over="ignore", invalid="ignore"):
                component_start = perf_counter() if profile_components else None
                if metrics is not None:
                    g, raw = estimator.gradient(theta[r], idx, cost, return_raw_mean=True)
                    if profile_components:
                        components["gradient"] += perf_counter() - component_start
                        component_start = perf_counter()
                    diagonal = metrics[r].update(raw)
                    cost.count("metric_updates")
                    if profile_components:
                        components["adaptive_update"] += perf_counter() - component_start
                else:
                    g = estimator.gradient(theta[r], idx, cost)
                    diagonal = frozen_diagonal if preconditioner == "frozen" else 1.0
                    if profile_components:
                        components["gradient"] += perf_counter() - component_start
                component_start = perf_counter() if profile_components else None
                if preconditioner == "frozen_full":
                    proposal = (theta[r] - 0.5 * step_size * (frozen_matrix @ g) +
                                np.sqrt(step_size) * (full_cholesky @ z))
                    cost.count("dense_metric_vector_products", 2)
                else:
                    proposal = theta[r] - 0.5 * step_size * diagonal * g + np.sqrt(step_size * diagonal) * z
                if profile_components:
                    components["drift_and_noise"] += perf_counter() - component_start
            cost.count("transitions")
            cost.count("normal_variates", estimator.model.d)
            if not np.isfinite(proposal).all():
                live[r] = False
                theta[r] = np.nan
                failures.append({"replica": r, "iteration": k, "reason": "nonfinite transition"})
            elif (max_standardized_radius is not None and
                  np.linalg.norm((proposal - failure_center) @ failure_transform) > max_standardized_radius):
                live[r] = False
                theta[r] = np.nan
                failures.append({"replica": r, "iteration": k, "reason": "frozen radius limit"})
            else:
                theta[r] = proposal
        if k == burn_in:
            burn_seconds = perf_counter() - start - functional_seconds
            burn_operations = dict(cost.operations)
        if functionals is not None and k > functional_burn_in:
            scoring_start = perf_counter()
            functional_sums += functionals(theta)
            functional_count += 1
            functional_seconds += perf_counter() - scoring_start
        if next_checkpoint < len(checkpoints) and k == checkpoints[next_checkpoint]:
            states[next_checkpoint] = theta
            if functionals is not None:
                functional_averages.append(functional_sums / functional_count if functional_count else
                                           np.full_like(functional_sums, np.nan))
            if cost_snapshots or k == steps:
                snapshots.append(snapshot(k))
            next_checkpoint += 1
    return ReplicaRun(checkpoints, states, snapshots, failures, {
        "method": estimator.method, "step_size": step_size, "batch_size": batch_size,
        "replicas": replicas, "seed": seed, "burn_in": burn_in,
        "sampling": "with replacement" if replace else "without replacement",
        "timing_scope": "sequential entire replica collection; setup charged once",
        "cache_bytes_excluding_model": estimator.cache_bytes,
        "cost_snapshot_scope": "all checkpoints" if cost_snapshots else "start and final only",
        "preconditioner": preconditioner, "alpha": alpha, "epsilon": epsilon,
        "Gamma": "omitted in practical adaptive Li variant; zero for fixed metrics",
        "metric_initialization": "supplied pilot" if initial_v is not None else "zero",
        "step_convention": "h_code = 2 * h_theory; drift h_code/2, noise sqrt(h_code)",
        "functional_evaluation_seconds_excluded": functional_seconds,
        "functional_burn_in": functional_burn_in,
        "max_standardized_radius": max_standardized_radius,
        "profile_components": profile_components,
        "component_timing_scope": "nested subset of sampler elapsed; not additional cost",
        "dtype": "float64", "transition": "theta - h*P*g/2 + sqrt(h*P)*z"},
        np.asarray(functional_averages) if functionals is not None else None,
        np.stack([m.v for m in metrics]) if metrics is not None and record_adaptive_state else None)


def run_chains(estimator, *, initial, step_size, batch_size, warmup, draws, chains, seed, **metric_options):
    """Return unthinned post-warmup draws and setup-inclusive collection costs.

    Diagnostics are computed separately, outside sampler timers.
    """
    if not isinstance(draws, int) or draws < 1 or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("positive draws and nonnegative warmup required")
    run = run_replicas(estimator, initial=initial, step_size=step_size,
                       batch_size=batch_size, steps=warmup + draws, replicas=chains,
                       seed=seed, burn_in=warmup,
                       checkpoints=np.arange(warmup + 1, warmup + draws + 1),
                       cost_snapshots=False, **metric_options)
    samples = run.states[run.checkpoints > warmup].transpose(1, 0, 2)
    return samples, run


def run_with_theory_step(estimator, *, h, **options):
    """Public h convention: theta-h*P*g+sqrt(2h)*chol(P)*z.

    Legacy step_size remains h_code for historical reproducibility. Supplying
    both conventions is an error, not an implicit override.
    """
    if "step_size" in options:
        raise ValueError("use h or legacy step_size, never both")
    run = run_replicas(estimator, step_size=2*h, **options)
    run.metadata.update(h=float(h), h_code=float(2*h),
                        public_transition="theta - h*P*g + sqrt(2*h)*chol(P)*z")
    return run

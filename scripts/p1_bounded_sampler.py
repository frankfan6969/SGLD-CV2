"""Frozen-core-equivalent sequential sampler with explicit wall-budget stopping.

No change to the P0/P1 locked core. Budget checks occur after whole-ensemble
iterations. Scored states exist only at prespecified checkpoints; terminal
work between checkpoints is retained as search cost, never scored as a prefix.
"""
from time import perf_counter
import numpy as np

from sgld_cv.preconditioners import RMSprop
from sgld_cv.randomness import generator, BATCH, NOISE


def bounded_replicas(estimator, *, initial, h, batch_size, replicas, steps, seed,
                     checkpoints, functionals, transform, max_radius,
                     matrix=None, metric_cost=None, alpha=None, epsilon=None,
                     max_sampler_seconds=120., stage_deadline=float("inf"), clock=perf_counter):
    if not np.isfinite(h) or h <= 0 or replicas < 1 or steps < 1:
        raise ValueError("invalid transition parameters")
    checkpoints = sorted(set(checkpoints))
    if not checkpoints or checkpoints[0] < 1 or checkpoints[-1] != steps:
        raise ValueError("positive prespecified checkpoints must end at steps")
    if not 1 <= batch_size <= estimator.model.N:
        raise ValueError("invalid batch size")
    d = estimator.model.d
    initial = np.asarray(initial, dtype=float)
    theta = np.tile(initial, (replicas, 1))
    if initial.shape != (d,) or transform.shape != (d,d):
        raise ValueError("invalid initial/whitening dimensions")
    cost = estimator.setup_cost.copy()
    if metric_cost is not None:
        cost.add(metric_cost)
    cholesky = None
    if matrix is not None:
        with cost.measure("full_metric_factorization"):
            cholesky = np.linalg.cholesky(matrix)
            cost.count("cholesky_factorizations")
    metrics = None
    if alpha is not None:
        with cost.measure("adaptive_state_initialization"):
            metrics = [RMSprop(d, alpha=alpha, epsilon=epsilon) for _ in range(replicas)]
            cost.count("adaptive_state_initializations", replicas)
            cost.count("adaptive_state_scalars", replicas*d)
    setup_seconds = cost.snapshot()["total_seconds"]
    setup_operations = dict(cost.operations)
    start = clock()
    batch_rng = [generator(seed,r,BATCH) for r in range(replicas)]
    noise_rng = [generator(seed,r,NOISE) for r in range(replicas)]
    functional_seconds = 0.
    t = clock(); sums = np.zeros_like(functionals(theta)); functional_seconds += clock()-t
    live = np.ones(replicas, dtype=bool)
    failures, saved = [], []
    reason, completed = "complete", 0

    def snapshot(k):
        current = cost.snapshot()
        current["seconds"]["burn_in"] = 0.
        current["seconds"]["sampling"] = max(0., clock()-start-functional_seconds)
        current["total_seconds"] = sum(current["seconds"].values())
        current["phase_operations"]["sampling"] = {
            key:value-setup_operations.get(key,0) for key,value in cost.operations.items()
            if value != setup_operations.get(key,0)}
        current["iteration"] = k
        return current

    for k in range(1,steps+1):
        if clock() >= stage_deadline:
            reason = "stage_budget_censored"; break
        if snapshot(completed)["total_seconds"] >= max_sampler_seconds:
            reason = "candidate_budget_censored"; break
        for r in range(replicas):
            if not live[r]:
                continue
            idx = None if estimator.method == "ula" else batch_rng[r].choice(estimator.model.N, batch_size, replace=False)
            z = noise_rng[r].standard_normal(d)
            with np.errstate(over="ignore", invalid="ignore"):
                if metrics is not None:
                    gradient, raw = estimator.gradient(theta[r],idx,cost,return_raw_mean=True)
                    diagonal = metrics[r].update(raw)
                    cost.count("metric_updates")
                else:
                    gradient = estimator.gradient(theta[r],idx,cost)
                    diagonal = 1.
                if matrix is None:
                    proposal = theta[r] - h*diagonal*gradient + np.sqrt(2*h*diagonal)*z
                else:
                    proposal = theta[r] - h*(matrix@gradient) + np.sqrt(2*h)*(cholesky@z)
                    cost.count("dense_metric_vector_products",2)
                radius = np.linalg.norm((proposal-initial)@transform)
            cost.count("transitions"); cost.count("normal_variates",d)
            if not np.isfinite(proposal).all() or not np.isfinite(radius) or radius > max_radius:
                failures.append({"replica":r,"iteration":k,"reason":"nonfinite transition" if not np.isfinite(proposal).all() else "frozen radius limit"})
                theta[r] = np.nan; live[r] = False
            else:
                theta[r] = proposal
        completed = k
        t = clock()
        with np.errstate(over="ignore", invalid="ignore"):
            sums += functionals(theta)
        functional_seconds += clock()-t
        if k in checkpoints:
            saved.append({"iteration":k,"endpoints":theta.copy(),"functional_averages":sums/k,
                          "cost":snapshot(k),"valid":bool(live.all())})
        if not live.any():
            reason = "all_replicas_failed"; break
    terminal_cost = snapshot(completed)
    for name, value in terminal_cost["operations"].items():
        if sum(phase.get(name,0) for phase in terminal_cost["phase_operations"].values()) != value:
            raise AssertionError("operation ledger does not partition exactly")
    return {"saved":saved,"terminal_state":theta,"terminal_functional_sums":sums,
            "terminal_cost":terminal_cost,"failures":failures,"stop_reason":reason,
            "completed_iterations":completed,"functional_evaluation_seconds":functional_seconds,
            "adaptive_final_v":np.stack([metric.v for metric in metrics]) if metrics is not None else np.empty((0,d)),
            "setup_seconds":setup_seconds,"cache_bytes":estimator.cache_bytes,
            "timing_scope":"sequential full replica ensemble; setup once; functional evaluation excluded",
            "budget_granularity":"complete ensemble iteration; possible overrun <= one iteration"}

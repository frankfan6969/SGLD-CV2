"""Execute phase-one correctness checks and freeze reproducible target assets.

This runs only the small validation grid, never the phase-two frontier search.
Existing output directories are refused. Verification mode does not run samplers.
"""
import argparse
from itertools import combinations
import json
from pathlib import Path
import platform
from time import perf_counter
import importlib.metadata

import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits, threadpool_info

from sgld_cv.anchors import find_anchor, supplied_anchor
from sgld_cv.costs import CostLedger
from sgld_cv.data import german_credit, synthetic_logistic
from sgld_cv.diagnostics import gradient_audit, chain_diagnostics
from sgld_cv.estimators import prepare_estimator
from sgld_cv.functionals import FunctionalSpec, functional_rmse
from sgld_cv.gaussian import make_gaussian_target
from sgld_cv.models import LogisticRegression
from sgld_cv.preconditioners import full_hessian_metric, fixed_anchor_metric
from sgld_cv.protocol import validate_protocol, make_lock, verify_lock, historical_seed_inventory
from sgld_cv.randomness import generator
from sgld_cv.references import hmc_reference, validate_reference_pair
from sgld_cv.samplers import run_replicas
from sgld_cv.transport import AccuracyMetric, empirical_w2
from sgld_cv.workflow import save_json, save_csv


ROOT = Path(__file__).resolve().parents[1]


def formula_checks():
    rng = np.random.default_rng(141900)
    model = LogisticRegression(rng.normal(size=(7, 3)), [0, 1, 1, 0, 0, 1, 0], .7)
    anchor = supplied_anchor([.1, -.2, .3], provenance="nonoptimal algebra-check anchor")
    theta = np.array([.3, .2, -.1])
    result = {}
    for method in ("sgld", "cv1", "cv2", "ula"):
        est = prepare_estimator(model, method, anchor)
        values = [est.gradient(theta, np.asarray(b)) for b in combinations(range(7), 2)]
        error = float(np.max(np.abs(np.mean(values, axis=0) - model.full_gradient(theta))))
        if error > 1e-11:
            raise AssertionError(f"biased {method}")
        result[method] = {"exact_enumeration_max_error": error, "cache_bytes": est.cache_bytes,
                          "cache_shapes": {name: list(getattr(est, name).shape)
                                           for name in ("theta_hat", "residual_hat", "gradient_hat", "weights_hat", "hessian_hat")
                                           if getattr(est, name) is not None}}
        if method in ("cv1", "cv2"):
            np.testing.assert_allclose(est.gradient(anchor.theta, [0]), model.full_gradient(anchor.theta), atol=1e-12)
    eps = 1e-5
    fd = [(model.potential(theta + e) - model.potential(theta - e))/(2*eps) for e in eps*np.eye(3)]
    np.testing.assert_allclose(fd, model.full_gradient(theta), atol=1e-8)
    result["derivative_max_error"] = float(np.max(np.abs(fd-model.full_gradient(theta))))
    return result


def build_target(name, c, out):
    v = c["validation"]
    if name == "gaussian":
        target = make_gaussian_target(N=v["N"], d=v["d"], kappa=v["gaussian_kappa"], seed=c["seeds"]["dataset"][0])
        model = target.model
        anchor = supplied_anchor(np.zeros(model.d), provenance="known constructed Gaussian mode")
        first = generator(c["seeds"]["new_reference"][0], 0, 4).multivariate_normal(
            np.zeros(model.d), target.covariance, size=(4, 1024))
        second = generator(c["seeds"]["new_reference"][1], 0, 4).multivariate_normal(
            np.zeros(model.d), target.covariance, size=(4, 2048))
        meta = {"reliable": True, "kind": "exact independent Gaussian draws", "covariance": target.covariance}
    elif name == "synthetic_logistic":
        model, source = synthetic_logistic(N=v["N"], d=v["d"], design_kappa=v["logistic_design_kappa"],
                                          seed=c["seeds"]["dataset"][0])
        anchor = find_anchor(model)
        ref = c["reference"][name]
        kwargs = {k: ref[k] for k in ("chains", "warmup", "step_size", "leapfrog_steps")}
        first, d1 = hmc_reference(model, anchor, seed=c["seeds"]["new_reference"][0], draws=ref["first_draws"], **kwargs)
        second, d2 = hmc_reference(model, anchor, seed=c["seeds"]["new_reference"][1], draws=ref["second_draws"], **kwargs)
        agreement = validate_reference_pair(first, second, d1, d2)
        meta = {"reliable": agreement["reliable"], "kind": "independently validated approximate HMC",
                "first": d1, "second": d2, "agreement": agreement, "source": source}
    elif name == "german":
        spec = c["reference"][name]
        model, source = german_credit(ROOT / spec["archive"])
        anchor = find_anchor(model)
        ref_folder = ROOT / spec["folder"]
        meta = json.loads((ref_folder / "reference_diagnostics.json").read_text())
        if [meta["first"]["seed"], meta["longer"]["seed"]] != spec["seeds"]:
            raise ValueError("German reference seeds changed")
        meta["source"] = source
        with np.load(ref_folder / "reference_chains.npz") as data:
            first, second = data["first"], data["longer"]
    else:
        raise ValueError(f"unknown validation target: {name}")
    save_json(out / "reference_diagnostics.json", meta)
    if not meta["reliable"]:
        raise RuntimeError(f"{name}: reference validation failed; no sampler ranking allowed")
    gate = c["reference"]["hmc_gate"]
    if name != "gaussian":
        for diagnostic in (meta["first"], meta.get("second", meta.get("longer"))):
            if not (diagnostic["max_rhat"] < gate["max_rhat_exclusive"] and
                    diagnostic["min_bulk_ess"] >= gate["min_bulk_ess"] and
                    diagnostic["min_tail_ess"] >= gate["min_tail_ess"] and
                    sum(diagnostic["divergences"]) <= gate["max_divergences"] and
                    min(diagnostic["acceptance"]) > gate["min_acceptance_exclusive"]):
                raise RuntimeError(f"{name}: frozen reference gate failed")
    H = model.full_hessian(anchor.theta)
    prediction_indices = (generator(c["seeds"]["prediction_rows"][0], 0, 1).choice(model.N, 8, replace=False)
                          if name != "gaussian" else np.array([], dtype=int))
    f = FunctionalSpec.from_geometry(anchor.theta, H, model.X[prediction_indices])
    calibration = first.reshape(-1, model.d)
    scoring = second.reshape(-1, model.d)
    calibration_values, scoring_values = f(calibration), f(scoring)
    target_raw = scoring_values.mean(axis=0)
    target_values, names = f.derive(target_raw)
    # Influence-function scales for posterior variance, frozen on first reference.
    scale_values = calibration_values.copy()
    d = model.d
    scale_values[:, d:2*d] = (calibration_values[:, :d]-calibration_values[:, :d].mean(axis=0))**2
    scale_values[:, 2*d+1] = (calibration_values[:, 2*d]-calibration_values[:, 2*d].mean())**2
    scales = np.maximum(scale_values.std(axis=0, ddof=1), 1e-8)
    if name == "gaussian":
        # Analytic expectations/scales replace Monte Carlo truth for this control.
        target_raw = np.r_[np.zeros(d), np.ones(d), 0., 1.]
        target_values, names = f.derive(target_raw)
        scales = np.r_[np.ones(d), np.full(d, np.sqrt(2)), 1., np.sqrt(2)]
    calibration_target, _ = f.derive(calibration_values.mean(axis=0))
    mcse = {}
    if name != "gaussian":
        import arviz as az
        for label, samples in (("calibration", first), ("scoring", second)):
            raw = f(samples)
            means = raw.mean(axis=(0, 1))
            influence = raw.copy()
            influence[..., d:2*d] -= 2 * means[:d] * raw[..., :d]
            influence[..., 2*d+1] -= 2 * means[2*d] * raw[..., 2*d]
            mcse[label] = [float(np.asarray(az.mcse(influence[..., j], method="mean")).item())
                           for j in range(influence.shape[-1])]
    save_json(out / "functional_reference.json", {"names": names, "target": target_values, "scales": scales,
        "independent_reference_disagreement_normalized": (calibration_target-target_values)/scales,
        "functional_mcse": mcse, "kind": meta["kind"],
        "limitations": "HMC pools remain correlated; uncertainty conditional on fixed data/reference; no floor subtraction"})
    directions = generator(c["seeds"]["projections"][0], 0, 1).normal(size=(d, c["accuracy"]["projections"]))
    directions /= np.linalg.norm(directions, axis=0)
    metric = AccuracyMetric("standardized_sliced_w2", anchor.theta.copy(), f.transform.copy(), directions)
    np.savez_compressed(out / "frozen_assets.npz", X=model.X, y=model.y, anchor=anchor.theta, hessian=H,
        reference_calibration=first, reference_scoring=second, projections=directions, transform=f.transform,
        slow_direction=f.slow_direction, prediction_indices=prediction_indices, function_targets=target_values,
        function_scales=scales)
    return model, anchor, H, f, metric, target_values, scales, names, calibration, scoring


def assert_costs(run, estimator, metric_cost, c, arm):
    v, model = c["validation"], estimator.model
    R, K, b = v["replicas"], v["steps"], v["batch_size"]
    setup = dict(estimator.setup_cost.operations)
    for key, value in metric_cost.operations.items():
        setup[key] = setup.get(key, 0) + value
    final = run.costs[-1]
    ops = final["operations"]
    expected = setup.get("observation_gradient_evaluations", 0) + R*K*(model.N if estimator.method == "ula" else b)
    if ops["observation_gradient_evaluations"] != expected:
        raise AssertionError("gradient accounting mismatch")
    if ops["transitions"] != R*K or ops["normal_variates"] != R*K*model.d:
        raise AssertionError("replica/transition accounting mismatch")
    if ops.get("dense_hessian_vector_products", 0) != (R*K if estimator.method == "cv2" else 0):
        raise AssertionError("CV2 dense HVP accounting mismatch")
    if ops.get("metric_updates", 0) != (R*K if arm["metric"] == "rmsprop" else 0):
        raise AssertionError("adaptive metric accounting mismatch")
    if ops.get("dense_metric_vector_products", 0) != (2*R*K if arm["metric"] == "full_hessian" else 0):
        raise AssertionError("full metric accounting mismatch")
    for key, value in ops.items():
        attributed = sum(phase.get(key, 0) for phase in final["phase_operations"].values())
        if attributed != value:
            raise AssertionError(f"unattributed or double-counted operation: {key}")
    return {"expected_gradients": expected, "observed_gradients": ops["observation_gradient_evaluations"],
            "phase_partition_exact": True}


def validate_target(name, built, c, out):
    model, anchor, H, f, metric, target, scales, names, calibration, scoring = built
    v = c["validation"]
    estimators = {m: prepare_estimator(model, m, anchor) for m in ("sgld", "cv1", "cv2", "ula")}
    audit_states = calibration[generator(c["seeds"]["audit"][0], 0, 1).choice(len(calibration), v["audit_states"], replace=False)]
    rows, errors = gradient_audit(estimators, audit_states, batch_size=v["batch_size"],
        batches=v["audit_batches"], seed=c["seeds"]["audit"][1], metric_matrix=np.linalg.inv(H))
    save_csv(out / "gradient_mse.csv", rows)
    np.savez_compressed(out / "gradient_errors.npz", states=audit_states, **errors)
    all_scores, rmse_rows, cost_rows, baselines, checks = [], [], [], [], []
    # All preparation/timing below is one-thread; shared anchor cost attributed per arm.
    for seed in c["seeds"]["validation"]:
        reference = scoring[generator(seed, 0, 84).choice(len(scoring), c["accuracy"]["reference_size"], replace=False)]
        for repeat in range(c["accuracy"]["reference_baselines"]):
            a = calibration[generator(seed, repeat, 85).choice(len(calibration), v["replicas"], replace=False)]
            baselines.append({"seed": seed, "repeat": repeat, "sw2": metric.distance(a, reference),
                              "interpretation": "resolution comparison, not a lower bound"})
        gaussian_controls = {}
        for label in generator(seed, 0, 7).permutation(list(c["arms"])):
            arm = c["arms"][str(label)]
            est = prepare_estimator(model, arm["method"], anchor)
            metric_cost, options = CostLedger(), {}
            P = np.eye(model.d)
            if arm["metric"] == "full_hessian":
                if c["version"] >= 2:
                    P, metric_cost, _ = fixed_anchor_metric(model, anchor, est, kind="full", **c["fixed_metric_regularization"])
                else:
                    P, metric_cost = full_hessian_metric(model, anchor, est)
                options = {"preconditioner": "frozen_full", "frozen_matrix": P, "metric_cost": metric_cost}
                h = v["step_fraction"]
            elif arm["metric"] == "fixed_diagonal":
                diagonal, metric_cost, _ = fixed_anchor_metric(model, anchor, est, kind="diagonal", **c["fixed_metric_regularization"])
                P = np.diag(diagonal)
                options = {"preconditioner": "frozen", "frozen_diagonal": diagonal, "metric_cost": metric_cost}
                curvature = np.linalg.eigvalsh(np.sqrt(diagonal)[:, None]*H*np.sqrt(diagonal)[None, :])[-1]
                h = v["step_fraction"]/curvature
            elif arm["metric"] == "rmsprop":
                a = c["adaptive_baseline"]
                options = {"preconditioner": "rmsprop", "alpha": a["alpha"], "epsilon": a["epsilon"]}
                h = v["adaptive_step_fraction"] / np.linalg.eigvalsh(H)[-1]
            else:
                h = v["step_fraction"] / np.linalg.eigvalsh(H)[-1]
            if c["version"] >= 2:
                h *= 2  # v2 fractions use the public/theory h convention
                options.update(profile_components=True, record_adaptive_state=True)
            trajectory = run_replicas(est, initial=anchor.theta, step_size=float(h), batch_size=v["batch_size"],
                steps=v["steps"], replicas=v["replicas"], seed=seed, checkpoints=v["checkpoints"],
                functionals=f, functional_burn_in=c["functionals"]["burn_in"],
                failure_center=anchor.theta, failure_transform=f.transform,
                max_standardized_radius=c["failure"]["standardized_radius_multiplier"]*np.sqrt(model.d), **options)
            if trajectory.failures:
                save_json(out / f"failures_{label}_{seed}.json", trajectory.failures)
                raise AssertionError(f"{name}/{label}: small validation run failed; protocol retained without retuning")
            np.testing.assert_array_equal(trajectory.states[0], np.tile(anchor.theta, (v["replicas"], 1)))
            if np.array_equal(trajectory.states[-1, 0], trajectory.states[-1, 1]):
                raise AssertionError("replicas have identical noise streams")
            check = assert_costs(trajectory, est, metric_cost, c, arm)
            checks.append({"arm": label, "seed": seed, **check})
            if name == "gaussian" and arm["method"] in ("cv2", "ula") and arm["metric"] != "rmsprop":
                gaussian_controls[(arm["metric"], arm["method"])] = trajectory.states
            np.savez_compressed(out / f"run_{label}_{seed}.npz", checkpoints=trajectory.checkpoints,
                endpoints=trajectory.states, functional_averages=trajectory.functional_averages,
                adaptive_final_v=trajectory.adaptive_final_v if trajectory.adaptive_final_v is not None else np.empty((0, model.d)))
            save_json(out / f"cost_{label}_{seed}.json", {"snapshots": trajectory.costs, "metadata": trajectory.metadata,
                                                       "failures": trajectory.failures})
            for k, states, averages, cost in zip(trajectory.checkpoints, trajectory.states, trajectory.functional_averages, trajectory.costs):
                all_scores.append({"arm": label, "seed": seed, "iteration": int(k), "h_code": h, "h_theory": h/2,
                    "sw2": metric.distance(states, reference),
                    "auxiliary_pot_w2": empirical_w2(states, reference) if model.d <= 3 else None,
                    "cost_scope": "entire sequential replica ensemble; setup once",
                    "total_seconds": cost["total_seconds"], "sampling_seconds": cost["seconds"]["sampling"]})
                if k > 0:
                    derived, _ = f.derive(averages)
                    functional_rows = functional_rmse(derived, target, scales, names)
                    rmse_rows.extend(dict(row, arm=label, seed=seed, iteration=int(k),
                        total_ensemble_seconds=cost["total_seconds"],
                        scope="replica RMSE; sequential ensemble cost; not per-chain time") for row in functional_rows)
            final = trajectory.costs[-1]
            for phase, counts in final["phase_operations"].items():
                for operation, count in counts.items():
                    cost_rows.append({"arm": label, "seed": seed, "phase": phase, "operation": operation, "count": count})
        if name == "gaussian":
            for geometry in (("identity", "full_hessian", "fixed_diagonal") if c["version"] >= 2 else ("identity", "full_hessian")):
                error = float(np.max(np.abs(gaussian_controls[(geometry, "cv2")]-gaussian_controls[(geometry, "ula")])))
                if error > 1e-10:
                    raise AssertionError("quadratic CV2/ULA paired paths differ")
                checks.append({"seed": seed, "geometry": geometry, "cv2_ula_max_path_error": error})
    save_csv(out / "endpoint_accuracy.csv", all_scores)
    save_csv(out / "functional_rmse.csv", rmse_rows)
    save_csv(out / "cost_operations_by_phase.csv", cost_rows)
    save_csv(out / "reference_baselines.csv", baselines)
    save_json(out / "checks.json", checks)
    return {"N": model.N, "d": model.d, "arms": len(c["arms"]), "seeds": len(c["seeds"]["validation"]),
            "check_count": len(checks), "status": "passed_correctness_only"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/phase1_protocol.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/phase1_validation")
    parser.add_argument("--verify-lock", type=Path)
    args = parser.parse_args()
    if args.verify_lock:
        verify_lock(json.loads(args.verify_lock.read_text()), ROOT)
        print("Frozen protocol, sources and target assets verified.")
        return
    c = validate_protocol(json.loads(args.protocol.read_text()))
    historical = historical_seed_inventory(ROOT)
    declared = {seed for seeds in c["seeds"].values() for seed in seeds}
    if declared.intersection(historical):
        raise ValueError(f"new seeds overlap historical inventory: {sorted(declared.intersection(historical))}")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "purpose": c["purpose"], "platform": platform.platform(),
        "versions": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "arviz", "POT", "threadpoolctl")},
        "validation_seeds_consumed": c["seeds"]["validation"],
        "reserved_pilot_selection_heldout_seeds_consumed": False}
    manifest["historical_seed_inventory_count"] = len(historical)
    manifest["new_seed_roles_disjoint_from_legacy_inventory"] = True
    save_json(args.output / "manifest.json", manifest)
    try:
        with threadpool_limits(limits=1):
            manifest["threadpools"] = threadpool_info()
            manifest["formula_checks"] = formula_checks()
            built, assets = {}, []
            for name in c["validation"]["targets"]:
                folder = args.output / name
                folder.mkdir()
                print(f"Preparing frozen {name} target/reference", flush=True)
                start = perf_counter()
                built[name] = build_target(name, c, folder)
                manifest.setdefault("reference_and_calibration_seconds", {})[name] = perf_counter()-start
                assets.extend(folder / p for p in ("reference_diagnostics.json", "functional_reference.json", "frozen_assets.npz"))
            spec = c["reference"]["german"]
            assets.extend([ROOT/spec["archive"], ROOT/spec["folder"]/"reference_chains.npz",
                           ROOT/spec["folder"]/"reference_diagnostics.json"])
            lock = make_lock(args.protocol, ROOT, assets)
            save_json(args.output / "protocol.lock.json", lock)
            manifest["targets"] = {}
            for name, target in built.items():
                print(f"Validating 11 arms on {name}; validation seeds only", flush=True)
                manifest["targets"][name] = validate_target(name, target, c, args.output/name)
            verify_lock(lock, ROOT)
            manifest.update(status="complete", interpretation="phase-one correctness only; not a frontier or publication result")
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save_json(args.output / "manifest.json", manifest)
    print(f"Phase-one validation complete: {args.output}", flush=True)


if __name__ == "__main__":
    main()

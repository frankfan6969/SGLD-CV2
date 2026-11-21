"""Complete revised-plan P0 without running a P1 frontier or consuming heldout seeds."""
import argparse
import json
from pathlib import Path
import shutil
import platform
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from validate_phase1 import validate_target, formula_checks
from sgld_cv.anchors import find_anchor, supplied_anchor
from sgld_cv.data import synthetic_logistic, german_credit
from sgld_cv.estimators import prepare_estimator
from sgld_cv.functionals import FunctionalSpec
from sgld_cv.gaussian import make_gaussian_target
from sgld_cv.models import GaussianRegression
from sgld_cv.preconditioners import fixed_anchor_metric, RMSprop
from sgld_cv.protocol import make_lock, verify_lock, validate_protocol, historical_seed_inventory, sha256
from sgld_cv.randomness import generator, BATCH
from sgld_cv.samplers import run_replicas, run_with_theory_step
from sgld_cv.transport import AccuracyMetric
from sgld_cv.uncertainty import replica_bootstrap, paired_bootstrap, hierarchical_paired_bootstrap, cost_views
from sgld_cv.workflow import save_json, save_csv

ROOT = Path(__file__).resolve().parents[1]


def load_target(name, c, output):
    source = ROOT/c["assets_source"]/name
    for filename in ("frozen_assets.npz", "functional_reference.json", "reference_diagnostics.json"):
        shutil.copyfile(source/filename, output/filename)
    with np.load(source/"frozen_assets.npz") as saved:
        assets = {k: saved[k] for k in saved.files}
    v = c["validation"]
    if name == "gaussian":
        target = make_gaussian_target(N=v["N"], d=v["d"], kappa=v["gaussian_kappa"], seed=c["seeds"]["dataset"][0])
        model = target.model
        anchor = supplied_anchor(assets["anchor"], provenance="known constructed Gaussian mode")
    elif name == "synthetic_logistic":
        model, _ = synthetic_logistic(N=v["N"], d=v["d"], design_kappa=v["logistic_design_kappa"], seed=c["seeds"]["dataset"][0])
        anchor = find_anchor(model)
    else:
        model, _ = german_credit(ROOT/c["reference"]["german"]["archive"])
        anchor = find_anchor(model)
    np.testing.assert_array_equal(model.X, assets["X"])
    np.testing.assert_array_equal(model.y, assets["y"])
    np.testing.assert_allclose(anchor.theta, assets["anchor"], rtol=0, atol=1e-10)
    H = model.full_hessian(anchor.theta)
    np.testing.assert_allclose(H, assets["hessian"], rtol=0, atol=1e-9)
    refmeta = json.loads((source/"reference_diagnostics.json").read_text())
    if not refmeta["reliable"]:
        raise ValueError("reused reference failed validation")
    f = FunctionalSpec(anchor.theta.copy(), assets["transform"], assets["slow_direction"], model.X[assets["prediction_indices"]])
    metric = AccuracyMetric("standardized_sliced_w2", anchor.theta.copy(), assets["transform"], assets["projections"])
    fmeta = json.loads((source/"functional_reference.json").read_text())
    return (model, anchor, H, f, metric, assets["function_targets"], assets["function_scales"], fmeta["names"],
            assets["reference_calibration"].reshape(-1, model.d), assets["reference_scoring"].reshape(-1, model.d))


def metric_and_li_checks(built, c, folder):
    model, anchor, H, f = built[:4]
    checks = []
    delta = .1*np.ones(model.d)
    est = prepare_estimator(model, "cv2", anchor)
    original_hessian = est.hessian_hat.copy()
    expected_gradient = est.gradient(anchor.theta+delta, np.arange(8))
    for kind in ("full", "diagonal"):
        P, metric_cost, metadata = fixed_anchor_metric(model, anchor, est, kind=kind, **c["fixed_metric_regularization"])
        np.testing.assert_array_equal(est.hessian_hat, original_hessian)
        np.testing.assert_array_equal(est.gradient(anchor.theta+delta, np.arange(8)), expected_gradient)
        checks.append({"check": "CV2_full_correction_unchanged", "kind": kind, "status": "passed", **metadata})
    # Multistep reproducibility, raw-gradient recurrence and independent adaptive states.
    seed = c["seeds"]["validation_checks"][0]
    for method in ("sgld", "cv1", "cv2"):
        est = prepare_estimator(model, method, anchor)
        for alpha in (.9, .99):
            for epsilon in (1e-5, 1e-3):
                options = dict(initial=anchor.theta, h=5e-7/np.linalg.eigvalsh(H)[-1], batch_size=16,
                    steps=24, replicas=4, seed=seed, checkpoints=list(range(25)), preconditioner="rmsprop",
                    alpha=alpha, epsilon=epsilon, record_adaptive_state=True,
                    profile_components=True)
                a = run_with_theory_step(est, **options)
                b = run_with_theory_step(est, **options)
                np.testing.assert_array_equal(a.states, b.states)
                np.testing.assert_array_equal(a.adaptive_final_v, b.adaptive_final_v)
                assert a.costs[-1]["operations"] == b.costs[-1]["operations"]
                assert not a.failures
                for r in range(4):
                    rng = generator(seed, r, BATCH)
                    expected_v = np.zeros(model.d)
                    for k in range(24):
                        idx = rng.choice(model.N, 16, replace=False)
                        raw = model.X[idx].T @ model.residual(a.states[k, r], idx)/16
                        expected_v = alpha*expected_v+(1-alpha)*raw**2
                    np.testing.assert_allclose(expected_v, a.adaptive_final_v[r], rtol=1e-13, atol=1e-15)
                ops = a.costs[-1]["operations"]
                assert ops["adaptive_state_initializations"] == 4
                assert ops["metric_updates"] == 4*24
                checks.append({"check": "Li_reproducibility_raw_recurrence_and_cost", "method": method,
                    "alpha": alpha, "epsilon": epsilon, "status": "passed",
                    "cost": cost_views(a.costs[-1]), "operations": ops})
    save_json(folder/"metric_li_checks.json", checks)
    return len(checks)


def gaussian_law_checks(c, folder):
    # Noncommuting geometry tests full P, not just the diagonal special case.
    model = GaussianRegression([[1., .5], [.3, 1.4], [1.2, -.2]], [0., 0., 0.], prior_precision=.7)
    anchor = supplied_anchor([0., 0.], provenance="Gaussian covariance-test mode")
    est = prepare_estimator(model, "ula", anchor)
    results = []
    for kind in ("full", "diagonal"):
        P, cost, meta = fixed_anchor_metric(model, anchor, est, kind=kind, **c["fixed_metric_regularization"])
        options = (dict(preconditioner="frozen_full", frozen_matrix=P) if kind == "full" else
                   dict(preconditioner="frozen", frozen_diagonal=P))
        matrix = P if kind == "full" else np.diag(P)
        h = .02
        run = run_with_theory_step(est, h=h, initial=anchor.theta, batch_size=2, steps=1,
            replicas=8192, seed=c["seeds"]["validation_checks"][0], metric_cost=cost, **options)
        measured = np.cov(run.states[-1], rowvar=False, ddof=0)
        expected = 2*h*matrix
        relative = float(np.linalg.norm(measured-expected)/np.linalg.norm(expected))
        if relative > .05:
            raise AssertionError("Gaussian one-step noise covariance test failed")
        # Recursive population law using the public h, checked against matrix update.
        Q = model.full_hessian(anchor.theta)
        A = np.eye(model.d)-h*matrix@Q
        covariance = np.zeros_like(Q)
        for _ in range(12):
            covariance = A@covariance@A.T+2*h*matrix
        results.append({"kind": kind, "noise_covariance_relative_error": relative,
                        "population_covariance_after_12_steps": covariance, **meta})
    save_json(folder/"gaussian_noise_checks.json", results)
    return results


def statistic_factory(reference, metric, target, scales, R):
    projections = metric.projections.shape[1]
    yp = np.sort(((reference-metric.center)@metric.transform)@metric.projections, axis=0)
    S = len(reference)
    knots = np.unique(np.r_[np.arange(R+1)/R, np.arange(S+1)/S])
    mid = (knots[1:]+knots[:-1])/2
    xi = np.minimum((mid*R).astype(int), R-1)
    yi = np.minimum((mid*S).astype(int), S-1)
    weights = np.diff(knots)
    def statistic(cloud):
        xp = np.sort(cloud[:, :projections], axis=0)
        sw2 = np.sqrt(np.mean(np.sum(weights[:, None]*(xp[xi]-yp[yi])**2, axis=0)))
        nrmse = np.sqrt(np.mean(((cloud[:, projections:]-target)/scales)**2, axis=0))
        return np.r_[sw2, nrmse]
    return statistic


def uncertainty_checks(built, c, folder):
    model, anchor, H, f, metric, target, scales, names, calibration, scoring = built
    u, R = c["uncertainty"], c["validation"]["replicas"]
    columns = ["sw2"]+["nrmse_"+name for name in names]
    clouds, stats, rows = {}, {}, []
    for s, seed in enumerate(c["seeds"]["validation"]):
        reference = scoring[generator(seed, 0, 84).choice(len(scoring), c["accuracy"]["reference_size"], replace=False)]
        stat = statistic_factory(reference, metric, target, scales, R)
        stats[s] = stat
        for arm in c["arms"]:
            with np.load(folder/f"run_{arm}_{seed}.npz") as run:
                projected = ((run["endpoints"][-1]-anchor.theta)@metric.transform)@metric.projections
                estimates, _ = f.derive(run["functional_averages"][-1])
                cloud = np.c_[projected, estimates]
            clouds[(s, arm)] = cloud
            interval = replica_bootstrap(cloud, stat, repeats=u["validation_repeats"],
                seed=c["seeds"]["uncertainty"][0]+s, level=u["level"])
            for j, name in enumerate(columns):
                rows.append({"arm": arm, "seed": seed, "metric": name,
                    "estimate": float(interval["estimate"][j]), "lower": float(interval["lower"][j]),
                    "upper": float(interval["upper"][j]), "scope": interval["scope"]})
    paired_rows = []
    for a, b in (("full_cv2", "full_cv1"), ("diagonal_cv2", "diagonal_cv1"), ("pcv2", "pcv1")):
        aa = np.stack([clouds[(s, a)] for s in range(len(stats))])
        bb = np.stack([clouds[(s, b)] for s in range(len(stats))])
        interval = hierarchical_paired_bootstrap(aa, bb, lambda s, x: stats[s](x),
            repeats=u["validation_repeats"], seed=c["seeds"]["uncertainty"][1], level=u["level"])
        for j, name in enumerate(columns):
            paired_rows.append({"first": a, "second": b, "metric": name,
                "estimate": float(interval["estimate"][j]), "lower": float(interval["lower"][j]),
                "upper": float(interval["upper"][j]), "scope": interval["scope"],
                "interpretation": "two-seed smoke; validates execution only, no superiority claim"})
    save_csv(folder/"replica_confidence_intervals.csv", rows)
    save_csv(folder/"paired_hierarchical_intervals.csv", paired_rows)
    return {"individual_interval_rows": len(rows), "paired_interval_rows": len(paired_rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT/"configs/p0_protocol_v2.json")
    parser.add_argument("--output", type=Path, default=ROOT/"results/p0_validation_v2")
    parser.add_argument("--verify-lock", type=Path)
    args = parser.parse_args()
    if args.verify_lock:
        verify_lock(json.loads(args.verify_lock.read_text()), ROOT)
        print("P0 lock verified.")
        return
    c = validate_protocol(json.loads(args.protocol.read_text()))
    verify_lock(json.loads((ROOT/c["assets_source"]/"snapshot.lock.json").read_text()), ROOT)
    used = historical_seed_inventory(ROOT)
    new = {s for role, seeds in c["seeds"].items() if role not in {"new_reference", "dataset", "projections", "prediction_rows"} for s in seeds}
    if used & new:
        raise ValueError("P0 new experimental seed roles overlap legacy seeds")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "protocol": str(args.protocol.relative_to(ROOT)), "platform": platform.platform(),
        "validation_seeds_consumed": c["seeds"]["validation"], "pilot_selection_heldout_consumed": False,
        "references": "reused verified frozen v1 assets, independent of new algorithm seeds",
        "scope": "P0 correctness, uncertainty and theory audit; no P1 frontier"}
    save_json(args.output/"manifest.json", manifest)
    try:
        with threadpool_limits(limits=1):
            manifest["formula_checks"] = formula_checks()
            built, assets = {}, []
            for name in c["validation"]["targets"]:
                folder = args.output/name
                folder.mkdir()
                built[name] = load_target(name, c, folder)
                assets.extend(folder/p for p in ("frozen_assets.npz", "functional_reference.json", "reference_diagnostics.json"))
            assets.extend([ROOT/"scripts/validate_p0.py", ROOT/"P0_THEORY_AUDIT.md", ROOT/"configs/p0_theory_claims.json",
                           ROOT/"SGLD_CV2_Updated_Experimental_Plan.md"])
            lock = make_lock(args.protocol, ROOT, assets)
            save_json(args.output/"protocol.lock.json", lock)
            manifest["targets"] = {}
            for name, target in built.items():
                print(f"P0: 14 arms, metric/Li checks and bootstrap on {name}", flush=True)
                folder = args.output/name
                result = validate_target(name, target, c, folder)
                result["metric_li_check_count"] = metric_and_li_checks(target, c, folder)
                result["uncertainty"] = uncertainty_checks(target, c, folder)
                manifest["targets"][name] = result
            manifest["gaussian_law_checks"] = gaussian_law_checks(c, args.output)
            verify_lock(lock, ROOT)
            manifest.update(status="complete", theory_claims="classified; original proof revisions quarantined for P5",
                            P1_empirical_comparisons_allowed=True, theorem_superiority_allowed=False)
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save_json(args.output/"manifest.json", manifest)
    print(f"P0 complete: {args.output}", flush=True)


if __name__ == "__main__":
    main()

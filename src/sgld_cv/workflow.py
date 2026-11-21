"""Six-method experiment orchestration, disjoint tuning/evaluation and artifacts."""
import csv
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from time import perf_counter
import numpy as np
from .anchors import find_anchor, supplied_anchor
from .data import synthetic_logistic, german_credit, year_model
from .gaussian import make_gaussian_target
from .estimators import prepare_estimator
from .preconditioners import metric_pilot
from .samplers import run_replicas, run_chains
from .references import ridge_posterior, CachedGaussianEstimator, hmc_reference, validate_reference_pair
from .diagnostics import gradient_audit, radius_audit, chain_diagnostics
from .transport import make_accuracy_metric, load_accuracy_metric, metric_fingerprint
from .randomness import generator

METHODS = ["sgld", "cv1", "cv2", "psgld", "pcv1", "pcv2"]


def checkpoint_within_budget(checkpoints, costs, budget_seconds):
    """Last positive checkpoint inside a common setup-inclusive time budget.

    The run must cross the budget so faster candidates are not scored early.
    """
    if budget_seconds is None:
        return len(checkpoints)-1
    if costs[-1]["total_seconds"] < budget_seconds:
        return None
    eligible=[i for i,(k,cost) in enumerate(zip(checkpoints,costs))
              if k>0 and cost["total_seconds"]<=budget_seconds]
    return eligible[-1] if eligible else None


def save_json(path, obj):
    def clean(value):
        if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)): return [clean(v) for v in value]
        if isinstance(value, np.ndarray): return clean(value.tolist())
        if isinstance(value, (np.integer,)): return int(value)
        if isinstance(value, (np.bool_,)): return bool(value)
        if isinstance(value, (float, np.floating)): return float(value) if np.isfinite(value) else None
        return value
    Path(path).write_text(json.dumps(clean(obj), indent=2, allow_nan=False) + "\n")


def save_csv(path, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader(); writer.writerows(rows)


def prepare_accuracy_metric(config, out, model, anchor, H, refmeta, *, frozen_selection=None):
    """Share one verified scoring calibration across tuning, probes and evaluation.

    Loading a frozen calibration deliberately bypasses estimation, even when a
    new HMC reference has different moments. Scored reference draws stay fresh.
    """
    spec = config.get("accuracy_metric", {})
    kind = spec.get("kind", "auto")
    if kind == "auto":
        kind = "w2" if model.d <= spec.get("cutoff", 10) else "standardized_sliced_w2"
    saved_meta = None
    if frozen_selection is not None:
        saved_meta = json.loads(Path(frozen_selection).read_text()).get("accuracy_metric")
        if saved_meta is not None and saved_meta.get("kind") != kind:
            raise ValueError("frozen tuning uses a different accuracy metric")
    if frozen_selection is not None and kind == "standardized_sliced_w2":
        if not saved_meta or not saved_meta.get("calibration_sha256"):
            raise ValueError("frozen scoring requires a hashed whitening/projection calibration")
        accuracy = load_accuracy_metric(
            Path(frozen_selection).with_name("accuracy_calibration.npz"),
            d=model.d, covariance_floor=saved_meta.get("covariance_floor"))
        if metric_fingerprint(accuracy) != saved_meta["calibration_sha256"]:
            raise ValueError("frozen accuracy calibration fingerprint mismatch")
        if accuracy.projections.shape[1] != spec.get("projections", 512):
            raise ValueError("frozen tuning uses a different accuracy metric projection count")
        source = "frozen pilot calibration"
    else:
        accuracy = make_accuracy_metric(model.d, kind=kind,
            center=np.asarray(refmeta.get("calibration_mean", anchor.theta)),
            covariance=(np.asarray(refmeta["calibration_covariance"])
                        if "calibration_covariance" in refmeta else np.linalg.inv(H)),
            projections=spec.get("projections", 512), seed=spec.get("seed", 926),
            eigenvalue_floor=spec.get("eigenvalue_floor", 1e-8))
        source = ("first independent HMC reference" if "calibration_mean" in refmeta
                  else "analytic Gaussian posterior")
    metadata = {"kind": accuracy.kind, "cutoff": spec.get("cutoff", 10),
        "projections": accuracy.projections.shape[1] if accuracy.projections is not None else None,
        "seed": spec.get("seed", 926), "covariance_floor": accuracy.covariance_floor,
        "eigenvalue_floor": spec.get("eigenvalue_floor", 1e-8),
        "calibration": source, "calibration_sha256": metric_fingerprint(accuracy)}
    if saved_meta is not None and any(metadata.get(key) != value
            for key, value in saved_meta.items() if key != "calibration"):
        raise ValueError("frozen tuning uses a different accuracy metric")
    if accuracy.projections is not None:
        np.savez_compressed(Path(out) / "accuracy_calibration.npz", center=accuracy.center,
                            transform=accuracy.transform, projections=accuracy.projections)
    return accuracy, metadata


def build_problem(config, out):
    spec = config["problem"]
    kind = spec["kind"]
    if kind == "gaussian":
        target = make_gaussian_target(**{k: v for k, v in spec.items() if k != "kind"})
        model, meta = target.model, target.metadata
        anchor = supplied_anchor(target.mean, provenance="known Gaussian mode")
        sample, second_sample = target.sample, target.sample
        posterior = None
        refmeta = {"kind": "exact Gaussian", "reliable": True}
    else:
        kwargs = {k: v for k, v in spec.items() if k != "kind"}
        if kind == "synthetic_logistic": model, meta = synthetic_logistic(**kwargs)
        elif kind == "german_credit": model, meta = german_credit(**kwargs)
        elif kind == "year_ridge": model, meta = year_model(**kwargs)
        else: raise ValueError(f"unknown problem {kind}")
        posterior = None
        if kind == "year_ridge":
            posterior = ridge_posterior(model)
            anchor = supplied_anchor(posterior.mean, provenance="analytic ridge mode", cost=posterior.cost)
            sample, second_sample = posterior.sample, posterior.sample
            refmeta = {"kind": "exact Gaussian ridge conditional on fixed noise variance", "reliable": True,
                       "noise_variance": model.noise_variance}
        else:
            anchor = find_anchor(model)
            options = config["reference"]
            print("Building two independent full-data HMC references...", flush=True)
            first, d1 = hmc_reference(model, anchor, **options)
            second, d2 = hmc_reference(model, anchor, **dict(options, seed=options["seed"]+1, draws=2*options["draws"]))
            agreement = validate_reference_pair(first, second, d1, d2)
            refmeta = {"kind": "approximate full-data HMC", "first": d1, "longer": d2,
                       "agreement": agreement, "reliable": agreement["reliable"]}
            refmeta["reference_mean"]=second.mean(axis=(0,1)).tolist()
            refmeta["reference_covariance"]=np.cov(second.reshape(-1,model.d),rowvar=False).tolist()
            # The first reference run calibrates whitening; the independent
            # second run supplies scored reference samples.
            refmeta["calibration_mean"]=first.mean(axis=(0,1)).tolist()
            refmeta["calibration_covariance"]=np.cov(first.reshape(-1,model.d),rowvar=False).tolist()
            np.savez_compressed(out / "reference_chains.npz", first=first, longer=second)
            def pool_sampler(pool):
                pool = pool.reshape(-1, model.d)
                def draw(size, rng):
                    if size > len(pool): raise ValueError("reference pool too small")
                    return pool[rng.choice(len(pool), size, replace=False)]
                return draw
            sample, second_sample = pool_sampler(first), pool_sampler(second)
    save_json(out / "reference_diagnostics.json", refmeta)
    H = posterior.precision if posterior is not None else model.full_hessian(anchor.theta)
    meta.update({"N": model.N, "d": model.d, "mode_hessian_kappa": float(np.linalg.cond(H)),
                 "prior_precision": model.prior_precision, "anchor": anchor.metadata})
    save_json(out / "problem.json", meta)
    np.savez_compressed(out / "geometry.npz", anchor=anchor.theta, hessian=H)
    return model, anchor, H, sample, second_sample, refmeta, posterior


def run_study(config, output, *, frozen_selection=None):
    """Pilot performs equal trajectory-count tuning, then held-out evaluation.

    Publication mode requires frozen settings and validated references. It does
    not silently re-tune when the evaluation seed changes.
    """
    out = Path(output)
    if set(config["tuning"]["seeds"]) & set(config["evaluation"]["seeds"]):
        raise ValueError("tuning and evaluation seeds must be disjoint")
    if config.get("purpose") == "publication" and frozen_selection is None:
        raise ValueError("publication runs require a frozen pilot selection")
    out.mkdir(parents=True, exist_ok=False)
    source_root = Path(__file__).parent
    manifest = {"status": "running", "config": config, "platform": platform.platform(),
                "versions": {k: importlib.metadata.version(k) for k in ["numpy", "scipy", "POT", "arviz"]},
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_root.glob("*.py")},
                "timing_scope": "sequential ensemble; setup once; evaluation outside samplers",
                "blas_threads": 1,
                "metric": "Li RMSprop raw likelihood mean; Gamma omitted; independent anchor-batch pilot initialization"}
    save_json(out / "manifest.json", manifest)
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            return _run(config, out, manifest, frozen_selection)
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save_json(out / "manifest.json", manifest)
        raise


def _run(c, out, manifest, frozen_selection):
    model, anchor, H, reference, validation_reference, refmeta, posterior = build_problem(c, out)
    if c.get("purpose") == "publication" and not refmeta["reliable"]:
        raise RuntimeError("reference failed validation; publication run blocked")
    reference_mean=np.asarray(refmeta.get("reference_mean",anchor.theta))
    reference_covariance=np.asarray(refmeta["reference_covariance"]) if "reference_covariance" in refmeta else np.linalg.inv(H)
    accuracy, manifest["accuracy_metric"] = prepare_accuracy_metric(
        c, out, model, anchor, H, refmeta, frozen_selection=frozen_selection)
    moment_transform=accuracy.transform if accuracy.transform is not None else np.linalg.cholesky(H)
    reference_covariance_white=moment_transform.T@reference_covariance@moment_transform
    tuning, evaluation = c["tuning"], c["evaluation"]
    metric_spec = c["metric"]
    batch_sizes = sorted(set(tuning["batch_sizes"]))
    if min(batch_sizes) < 1 or max(batch_sizes) > model.N: raise ValueError("invalid tuning batch size")
    metric_pilots = {b: metric_pilot(model, anchor.theta, batch_size=b, **metric_spec) for b in batch_sizes}
    save_json(out / "metric_pilots.json", {b: {"v": p.second_moment, "diagonal": p.diagonal,
                                              "cost": p.cost.snapshot(), "metadata": p.metadata} for b, p in metric_pilots.items()})
    def options(name, b, frozen=False):
        if not name.startswith("p"): return {}
        pilot = metric_pilots[b]
        common = {"metric_cost": pilot.cost, "alpha": metric_spec.get("alpha", .99),
                  "epsilon": metric_spec.get("epsilon", 1e-5)}
        if frozen: return dict(common, preconditioner="frozen", frozen_diagonal=pilot.diagonal)
        return dict(common, preconditioner="rmsprop", initial_v=pilot.second_moment)
    def estimator_for(name):
        return prepare_estimator(model, name[1:] if name.startswith("p") else name, anchor)
    def h_for(name, b, fraction):
        # Fixed pilot metric sets a common interpretable initial curvature scale.
        D = metric_pilots[b].diagonal if name.startswith("p") else np.ones(model.d)
        return float(fraction / np.linalg.eigvalsh(np.sqrt(D)[:, None] * H * np.sqrt(D)[None, :])[-1])
    data_hash = hashlib.sha256()
    for first in range(0, model.N, 8192):
        data_hash.update(model.X[first:first+8192].tobytes())
        data_hash.update(model.y[first:first+8192].tobytes())
    manifest["model_data_sha256"] = data_hash.hexdigest()
    signature = hashlib.sha256((json.dumps(c["problem"], sort_keys=True)+data_hash.hexdigest()).encode()).hexdigest()
    if frozen_selection is None and c.get("comparison") == "common":
        b = batch_sizes[0]
        fraction = tuning["step_fractions"][0]
        h = min(h_for(name,b,fraction) for name in METHODS)
        selection = {name: {"batch_size": b, "step_size": h} for name in METHODS}
        frozen = {"problem_signature": signature,"metric_spec":metric_spec,"selection":selection,
                  "pilot_seeds":tuning["seeds"],"criterion":"common h and b for all six methods",
                  "accuracy_metric":manifest["accuracy_metric"],
                  "inspected_evaluation_seeds":evaluation["seeds"],
                  "budget":"no method-specific tuning; common conservative initial curvature scale"}
    elif frozen_selection is None:
        trials = []
        budget_seconds=tuning.get("budget_seconds")
        if budget_seconds is not None and (not np.isfinite(budget_seconds) or budget_seconds <= 0):
            raise ValueError("tuning budget_seconds must be positive")
        probe_checkpoints=np.unique(np.r_[0,np.geomspace(1,tuning["steps"],20).astype(int),tuning["steps"]])
        precision_chol=np.linalg.cholesky(H)
        order = generator(tuning["seeds"][0], 0, 87).permutation(METHODS)
        for name in order:
            budget_label=(f"same setup-inclusive {budget_seconds:g}-second budget" if budget_seconds is not None
                          else "same candidate/replica/iteration budget")
            print(f"Tuning {name} ({budget_label})", flush=True)
            for b in batch_sizes:
                for fraction in tuning["step_fractions"]:
                    h = h_for(name, b, fraction)
                    for seed in tuning["seeds"]:
                        estimator = estimator_for(name)
                        run = run_replicas(estimator, initial=anchor.theta, step_size=h, batch_size=b,
                              steps=tuning["steps"], replicas=tuning["replicas"], seed=seed,
                              checkpoints=probe_checkpoints, **options(name,b))
                        ref = validation_reference(tuning["replicas"], generator(seed, 0, 84))
                        chosen=checkpoint_within_budget(run.checkpoints,run.costs,budget_seconds)
                        reached=budget_seconds is None or run.costs[-1]["total_seconds"]>=budget_seconds
                        stable=not run.failures and np.isfinite(run.states).all()
                        if stable:
                            radii=np.linalg.norm((run.states-anchor.theta)@precision_chol,axis=-1)
                            stable=bool(np.max(radii)<tuning.get("stability_limit",100*np.sqrt(model.d)))
                        try: w = accuracy.distance(run.states[chosen], ref) if stable and chosen is not None else np.inf
                        except (ValueError, FloatingPointError, RuntimeError, Warning): w = np.inf
                        trials.append({"method": name, "batch_size": b, "fraction": fraction,
                            "step_size": h, "seed": seed, "accuracy": w, "metric": accuracy.kind,
                            "scored_iteration":int(run.checkpoints[chosen]) if chosen is not None else None,
                            "scored_total_seconds":run.costs[chosen]["total_seconds"] if chosen is not None else None,
                            "budget_seconds":budget_seconds,"budget_reached":reached,"stable":stable,
                            "total_seconds": run.costs[-1]["total_seconds"]})
            save_csv(out / "tuning_trials.csv", trials)
        selection = {}
        for name in METHODS:
            candidates = []
            for b in batch_sizes:
                for f in tuning["step_fractions"]:
                    group = [t for t in trials if t["method"]==name and t["batch_size"]==b and t["fraction"]==f]
                    score = float(np.mean([t["accuracy"] for t in group]))
                    if np.isfinite(score): candidates.append((score, b, f))
            if not candidates: raise RuntimeError(f"all pilot candidates failed for {name}")
            score, b, f = min(candidates)
            selection[name] = {"batch_size": b, "step_size": h_for(name,b,f), "pilot_mean_accuracy": score}
        frozen = {"problem_signature": signature, "metric_spec": metric_spec, "selection": selection,
                  "pilot_seeds": tuning["seeds"],
                  "criterion": (f"minimum mean {accuracy.kind} at common measured time budget on tuning seeds"
                                if budget_seconds is not None else
                                f"minimum mean endpoint {accuracy.kind} on tuning seeds"),
                  "accuracy_metric":manifest["accuracy_metric"],
                  "inspected_evaluation_seeds":evaluation["seeds"],
                  "budget": (f"same measured setup-inclusive {budget_seconds} seconds per candidate/seed; "
                             "last recorded checkpoint within budget" if budget_seconds is not None else
                             "equal candidates, seeds, replicas and transitions; wall time recorded, not equalized")}
    else:
        frozen = json.loads(Path(frozen_selection).read_text())
        if frozen["problem_signature"] != signature or frozen["metric_spec"] != metric_spec:
            raise ValueError("frozen tuning belongs to a different target/metric")
        if set(frozen["pilot_seeds"]) & set(evaluation["seeds"]):
            raise ValueError("frozen pilot seeds overlap evaluation seeds")
        if c.get("purpose")=="publication" and set(frozen.get("inspected_evaluation_seeds",[])) & set(evaluation["seeds"]):
            raise ValueError("publication seeds overlap already inspected pilot evaluation seeds")
        if c.get("purpose")=="publication" and set(frozen.get("horizon_probe",{}).get("seeds",[])) & set(evaluation["seeds"]):
            raise ValueError("publication seeds overlap already inspected horizon probe seeds")
        if c.get("purpose")=="publication":
            if "final_horizon" not in frozen:
                raise ValueError("publication requires a verified frozen final_horizon")
            if evaluation["steps"]!=frozen["final_horizon"]:
                raise ValueError("publication evaluation steps must equal frozen final_horizon")
    selection = frozen["selection"]
    if set(selection)!=set(METHODS) or any(s["batch_size"] not in metric_pilots for s in selection.values()):
        raise ValueError("selection must contain all six methods and prepared batch sizes")
    save_json(out / "frozen_selection.json", frozen)
    metrics, efficiency, baseline_rows, audit_rows = [], [], [], []
    checkpoints = np.unique(np.r_[0, np.geomspace(1, evaluation["steps"], 15).astype(int), evaluation["steps"]])
    estimators = {m: prepare_estimator(model,m,anchor) for m in ["sgld","cv1","cv2"]}
    audit_states = validation_reference(c["audit"]["states"], generator(9100,0,91))
    for b in batch_sizes:
        summary, errors = gradient_audit(estimators, audit_states, batch_size=b, batches=c["audit"]["batches"], seed=9101)
        audit_rows.extend(summary)
        np.savez_compressed(out/f"gradient_errors_b{b}.npz", states=audit_states, **errors)
    save_csv(out / "gradient_mse.csv", audit_rows)
    eig, vectors = np.linalg.eigh(H)
    directions = np.vstack([vectors[:,0],vectors[:,-1],generator(9102,0,91).normal(size=(4,model.d))])
    save_csv(out / "gradient_radius.csv", radius_audit(estimators,anchor.theta,radii=c["audit"]["radii"],
              directions=directions,batch_size=batch_sizes[0],batches=c["audit"]["batches"],seed=9103))
    for seed in evaluation["seeds"]:
        ref = validation_reference(evaluation["reference_size"], generator(seed,0,84))
        np.save(out/f"reference_seed{seed}.npy",ref)
        for j in range(3):
            a = reference(evaluation["replicas"],generator(seed,j,85))
            b = validation_reference(evaluation["reference_size"],generator(seed,j,86))
            value=accuracy.distance(a,b)
            baseline_rows.append({"seed":seed,"repeat":j,"accuracy":value,
                                  "w2":value if accuracy.kind=="w2" else None,"metric":accuracy.kind})
        for name in generator(seed,0,87).permutation(METHODS):
            print(f"Evaluating seed={seed} {name}",flush=True)
            setting=selection[name]; b=setting["batch_size"]; h=setting["step_size"]
            estimator=estimator_for(name)
            run=run_replicas(estimator,initial=anchor.theta,step_size=h,batch_size=b,
                  steps=evaluation["steps"],replicas=evaluation["replicas"],seed=seed,checkpoints=checkpoints,**options(name,b))
            np.savez_compressed(out/f"replicas_{name}_seed{seed}.npz",states=run.states,checkpoints=run.checkpoints)
            save_json(out/f"costs_{name}_seed{seed}.json",{"costs":run.costs,"failures":run.failures,"metadata":run.metadata})
            for k,states,cost in zip(run.checkpoints,run.states,run.costs):
                try: w=accuracy.distance(states,ref);status="ok"
                except (ValueError,FloatingPointError,RuntimeError,Warning) as exc: w=None;status=str(exc)
                valid=np.isfinite(states).all()
                if valid:
                    sample_mean=states.mean(axis=0)
                    sample_cov=np.cov(states,rowvar=False,ddof=0)
                    mean_std=float(np.linalg.norm((sample_mean-reference_mean)@moment_transform)/np.sqrt(model.d))
                    covariance_std=float(np.linalg.norm(moment_transform.T@sample_cov@moment_transform-
                                                        reference_covariance_white)/model.d)
                metrics.append({"method":name,"seed":seed,"iteration":int(k),"batch_size":b,"step_size":h,
                   "accuracy":w,"metric":accuracy.kind,"w2":w if accuracy.kind=="w2" else None,
                   "status":status,"reference_reliable":refmeta["reliable"],"total_seconds":cost["total_seconds"],
                   "mean_error":float(np.linalg.norm(sample_mean-reference_mean)) if valid else None,
                   "mean_error_standardized":mean_std if valid else None,
                   "covariance_relative_error":float(np.linalg.norm(sample_cov-reference_covariance)/np.linalg.norm(reference_covariance)) if valid else None,
                   "covariance_error_standardized":covariance_std if valid else None,
                   "observation_gradients":cost["operations"].get("observation_gradient_evaluations",0)})
            # Main six methods plus separately labeled frozen-metric ESS sensitivity.
            for freeze in ([False,True] if name.startswith("p") else [False]):
                label=name+"_frozen" if freeze else name
                chain_seed=seed+100000
                # Shared dispersed starts across all methods, independent of reference states.
                transform=np.linalg.cholesky(np.linalg.inv(H))
                initial=anchor.theta+generator(chain_seed,0,83).normal(size=(c["chains"]["count"],model.d))@transform.T
                samples,chain_run=run_chains(estimator,initial=initial,step_size=h,batch_size=b,
                    warmup=c["chains"]["warmup"],draws=c["chains"]["draws"],chains=c["chains"]["count"],seed=chain_seed,
                    **options(name,b,frozen=freeze))
                diagnostic=chain_diagnostics(samples,cost=chain_run.costs[-1],adaptive=name.startswith("p") and not freeze)
                # Weak-curvature direction is an additional quantity, not a replacement for coordinate diagnostics.
                weak=chain_diagnostics((samples@vectors[:,0])[:,:,None],cost=chain_run.costs[-1],adaptive=name.startswith("p") and not freeze)
                np.save(out/f"chains_{label}_seed{seed}.npy",samples)
                save_json(out/f"diagnostics_{label}_seed{seed}.json",{"diagnostics":diagnostic,"weak_direction":weak,
                          "cost":chain_run.costs[-1],"failures":chain_run.failures})
                for row in diagnostic["coordinates"]:
                    efficiency.append(dict(row,method=label,seed=seed,reliable=diagnostic["reliable"],
                       reference_reliable=refmeta["reliable"],total_seconds=chain_run.costs[-1]["total_seconds"]))
        save_csv(out/"metrics.csv",metrics); save_csv(out/"ess.csv",efficiency)
        save_csv(out/"reference_baselines.csv",baseline_rows)
    if posterior is not None:
        start=perf_counter(); direct=posterior.sample(evaluation["replicas"],generator(9999,0,88));elapsed=perf_counter()-start
        np.save(out/"direct_gaussian.npy",direct)
        cached=CachedGaussianEstimator(model,posterior)
        h=float(selection["cv2"]["step_size"])
        control=run_replicas(cached,initial=anchor.theta,step_size=h,batch_size=batch_sizes[0],steps=evaluation["steps"],
             replicas=evaluation["replicas"],seed=evaluation["seeds"][0],checkpoints=checkpoints)
        np.savez_compressed(out/"cached_ula.npz",states=control.states,checkpoints=control.checkpoints)
        save_json(out/"ridge_controls.json",{"direct_sampling_seconds":elapsed,"setup":posterior.cost.snapshot(),
                     "direct_total_seconds":elapsed+sum(posterior.cost.seconds.values()),"cached_ula_costs":control.costs})
        control_reference=validation_reference(evaluation["reference_size"],generator(evaluation["seeds"][0],0,84))
        control_rows=[{"method":"direct_iid","iteration":0,"accuracy":accuracy.distance(direct,control_reference),"metric":accuracy.kind,
                       "total_seconds":elapsed+sum(posterior.cost.seconds.values())}]
        for k,states,cost in zip(control.checkpoints,control.states,control.costs):
            control_rows.append({"method":"cached_ula","iteration":int(k),"accuracy":accuracy.distance(states,control_reference),"metric":accuracy.kind,
                                  "total_seconds":cost["total_seconds"]})
        save_csv(out/"ridge_control_metrics.csv",control_rows)
    start_distances=[r["accuracy"] for r in metrics if r["iteration"]==0 and r["accuracy"] is not None]
    baseline_median=float(np.median([r["accuracy"] for r in baseline_rows]))
    manifest["transport_resolution"]={"reference_baseline_median":baseline_median,
       "initial_point_mass_distance_median":float(np.median(start_distances)),
       "baseline_exceeds_initial_distance":bool(baseline_median>=np.median(start_distances)),
       "warning":"Empirical reference-reference distance is a calibration, not a lower bound; collapsed distributions can score below it. Increase sample counts and check moments before ranking methods."}
    manifest.update(status="complete",reference_reliable=refmeta["reliable"],
                    scope="validation/pilot evidence unless purpose=publication and all diagnostics pass")
    save_json(out/"manifest.json",manifest)
    return out

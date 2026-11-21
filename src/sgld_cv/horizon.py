"""Rough and post-tuning horizon probes for one representative posterior."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .workflow import build_problem, prepare_accuracy_metric, METHODS, save_json, save_csv
from .preconditioners import metric_pilot
from .estimators import prepare_estimator
from .samplers import run_replicas
from .randomness import generator


def provisional_horizon(rows, *, tolerance=.10, window=3):
    """Return earliest sustained plateau across accuracy and posterior moments.

    This detects an order of magnitude only. A biased stationary distribution
    can plateau, so the result never certifies posterior accuracy or ESS.
    """
    if not 0 < tolerance < 1 or window < 2:
        raise ValueError("invalid plateau rule")
    ks=sorted({int(r["iteration"]) for r in rows if int(r["iteration"])>0})
    groups={k:[r for r in rows if int(r["iteration"])==k] for k in ks}
    if len(ks)<window or any(not all(r["status"]=="ok" for r in groups[k]) for k in ks):
        return None
    series=np.array([[np.mean([float(r[key]) for r in groups[k]])
                      for key in ("accuracy","mean_error_std","covariance_error_std")] for k in ks])
    tail=np.median(series[-window:],axis=0)
    # A modest absolute scale prevents tiny mean errors from demanding
    # unrealistic relative precision from a finite reference sample.
    scales=np.maximum(np.abs(tail),.1)
    for i in range(1,len(ks)-window+1):
        if np.all(np.abs(series[i:]-tail)<=tolerance*scales):
            return int(ks[i])
    return None


def run_horizon_probe(config, output, *, frozen_selection=None):
    out=Path(output)
    out.mkdir(parents=True,exist_ok=False)
    save_json(out/"run_config.json",config)
    spec=config["horizon_probe"]
    seeds=list(spec["seeds"])
    if not seeds or set(seeds)&(set(config["tuning"]["seeds"])|set(config["evaluation"]["seeds"])):
        raise ValueError("horizon seeds must be nonempty and disjoint from tuning/evaluation")
    steps=int(spec["max_steps"])
    checkpoints=np.unique(np.r_[0,np.array(spec.get("checkpoints",
        np.geomspace(1,steps,12).astype(int)),dtype=int),steps])
    if steps<1 or np.any(checkpoints<0) or np.any(checkpoints>steps):
        raise ValueError("invalid horizon checkpoints")
    model,anchor,H,reference,validation_reference,refmeta,posterior=build_problem(config,out)
    if config["problem"]["kind"] not in ("synthetic_logistic","german_credit"):
        raise ValueError("horizon probe is intended for representative logistic cases")
    metric, accuracy_metadata = prepare_accuracy_metric(
        config, out, model, anchor, H, refmeta, frozen_selection=frozen_selection)
    whitening=metric.transform if metric.transform is not None else np.linalg.cholesky(H)
    target_mean=np.asarray(refmeta["reference_mean"])
    target_cov=np.asarray(refmeta["reference_covariance"])
    target_cov_white=whitening.T@target_cov@whitening
    data_hash=hashlib.sha256()
    for first in range(0,model.N,8192):
        data_hash.update(model.X[first:first+8192].tobytes())
        data_hash.update(model.y[first:first+8192].tobytes())
    signature=hashlib.sha256((json.dumps(config["problem"],sort_keys=True)+data_hash.hexdigest()).encode()).hexdigest()
    frozen=None
    if frozen_selection is not None:
        frozen=json.loads(Path(frozen_selection).read_text())
        if frozen["problem_signature"]!=signature or frozen["metric_spec"]!=config["metric"]:
            raise ValueError("frozen selection belongs to another posterior or metric")
        if set(seeds)&(set(frozen["pilot_seeds"])|set(frozen.get("inspected_evaluation_seeds",[]))
                       |set(frozen.get("horizon_probe",{}).get("seeds",[]))):
            raise ValueError("horizon seeds overlap previously inspected seeds")
    batches=sorted({frozen["selection"][m]["batch_size"] for m in METHODS}) if frozen else [int(spec["batch_size"])]
    pilots={b:metric_pilot(model,anchor.theta,batch_size=b,**config["metric"]) for b in batches}
    rows=[];baseline=[]
    eigen_max={}
    for b in batches:
        for name in METHODS:
            D=pilots[b].diagonal if name.startswith("p") else np.ones(model.d)
            eigen_max[(b,name)]=float(np.linalg.eigvalsh(np.sqrt(D)[:,None]*H*np.sqrt(D)[None,:])[-1])
    ref_by_seed={}
    for seed in seeds:
        ref=validation_reference(spec["reference_size"],generator(seed,0,84))
        ref_by_seed[seed]=ref
        for repeat in range(3):
            other=reference(spec["reference_size"],generator(seed,repeat,85))
            baseline.append({"seed":seed,"repeat":repeat,"accuracy":metric.distance(other,ref)})
    for name in METHODS:
        b=frozen["selection"][name]["batch_size"] if frozen else batches[0]
        h=frozen["selection"][name]["step_size"] if frozen else float(spec["step_fraction"])/eigen_max[(b,name)]
        estimator=prepare_estimator(model,name[1:] if name.startswith("p") else name,anchor)
        options={}
        if name.startswith("p"):
            options=dict(preconditioner="rmsprop",initial_v=pilots[b].second_moment,
                         metric_cost=pilots[b].cost,alpha=config["metric"].get("alpha",.99),
                         epsilon=config["metric"].get("epsilon",1e-5))
        for seed in seeds:
            run=run_replicas(estimator,initial=anchor.theta,step_size=h,batch_size=b,
                steps=steps,replicas=spec["replicas"],seed=seed,checkpoints=checkpoints,**options)
            np.savez_compressed(out/f"probe_{name}_seed{seed}.npz",states=run.states,checkpoints=run.checkpoints)
            save_json(out/f"costs_{name}_seed{seed}.json",{"costs":run.costs,"failures":run.failures})
            for k,states,cost in zip(run.checkpoints,run.states,run.costs):
                try:
                    value=metric.distance(states,ref_by_seed[seed]);status="ok"
                    standardized_mean=float(np.linalg.norm((states.mean(axis=0)-target_mean)@whitening)/np.sqrt(model.d))
                    covariance=np.cov(states,rowvar=False,ddof=0)
                    covariance_white=whitening.T@covariance@whitening
                    covariance_error=float(np.linalg.norm(covariance_white-target_cov_white)/model.d)
                except (ValueError,FloatingPointError,RuntimeError,Warning) as exc:
                    value=standardized_mean=covariance_error=None;status=str(exc)
                rows.append({"method":name,"seed":seed,"iteration":int(k),"step_size":h,
                    "batch_size":b,"accuracy":value,"mean_error_std":standardized_mean,
                    "covariance_error_std":covariance_error,"total_seconds":cost["total_seconds"],"status":status})
        save_csv(out/"horizon_curves.csv",rows)
    summary={m:provisional_horizon([r for r in rows if r["method"]==m],
        tolerance=spec.get("plateau_tolerance",.1),window=spec.get("plateau_window",3)) for m in METHODS}
    recommended=max(summary.values()) if all(v is not None for v in summary.values()) else None
    floor=float(np.quantile([r["accuracy"] for r in baseline],.75))
    initial=float(np.median([r["accuracy"] for r in rows if r["iteration"]==0]))
    result={"phase":"post_tuning" if frozen else "rough","metric":metric.kind,
        "accuracy_metric":accuracy_metadata,
        "reference_reliable":refmeta["reliable"],"seed_list":seeds,"max_steps":steps,
        "provisional_k_by_method":summary,"provisional_common_k":recommended,
        "next_probe_max_steps":2*steps if recommended is None else None,
        "reference_baseline_q75":floor,"initial_point_mass_accuracy":initial,
        "resolution_warning":bool(floor>=initial),
        "model_data_sha256":data_hash.hexdigest(),
        "source_sha256":{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                         for name in ("horizon.py","workflow.py","transport.py","samplers.py")},
        "interpretation":("No sustained plateau confirmed within the current iteration budget; "
                          "K_needed is undetermined. This does not prove nonconvergence or K_needed > max_steps. "
                          if recommended is None else "") +
                         "Plateau gives an order-of-magnitude horizon only; verify ESS, posterior moments and bias with held-out runs."}
    save_csv(out/"horizon_baselines.csv",baseline)
    save_json(out/"horizon_summary.json",result)
    if frozen and recommended is not None and not result["resolution_warning"] and refmeta["reliable"]:
        protocol=dict(frozen,final_horizon=recommended,accuracy_metric=accuracy_metadata,
                      horizon_probe={"seeds":seeds,"max_steps":steps,"criterion":spec,
                                     "summary_sha256":hashlib.sha256((out/"horizon_summary.json").read_bytes()).hexdigest()})
        save_json(out/"frozen_protocol.json",protocol)
    return result

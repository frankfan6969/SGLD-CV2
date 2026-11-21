"""Diagnostic-only exact-stream replay; never tune, score, or change pilot files."""
from collections import deque
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

from sgld_cv.anchors import find_anchor
from sgld_cv.data import german_credit
from sgld_cv.estimators import prepare_estimator
from sgld_cv.preconditioners import fixed_anchor_metric, RMSprop
from sgld_cv.protocol import sha256, verify_lock
from sgld_cv.randomness import generator, BATCH, NOISE
from sgld_cv.workflow import save_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/p1_german_diagnostic_replay_v1"
PILOT = ROOT / "results/p1_german_pilot_v1"


def independent_components(model, anchor, theta, idx, method):
    """Direct per-observation formula, independent of estimator cache assembly."""
    X, y = model.X, model.y
    p0 = expit(X @ anchor)
    prior = model.prior_precision * theta
    base = np.sum(X * (p0-y)[:, None], axis=0)
    correction = np.zeros(model.d)
    remainder = np.zeros(model.d)
    if method == "ula":
        direct = np.sum(X * (expit(X @ theta)-y)[:, None], axis=0) + prior
    else:
        delta = theta-anchor
        residual = expit(X[idx] @ theta)-p0[idx]
        if method == "cv2":
            correction = np.sum(X * ((p0*(1-p0))*(X @ delta))[:, None], axis=0)
            residual -= p0[idx]*(1-p0[idx])*(X[idx] @ delta)
        remainder = (model.N/len(idx))*np.sum(X[idx]*residual[:, None], axis=0)
        direct = base+correction+remainder+prior
    return direct, base, prior, correction, remainder


def replay(model, anchor, candidate, seed, selected, transform):
    start = perf_counter()
    name = f"{candidate['id']}_seed{seed}"
    original = json.loads((PILOT/f"result_{name}.json").read_text())
    est = prepare_estimator(model,candidate["method"],anchor)
    P = L = None
    adaptive = candidate["geometry"] == "li2016"
    if not adaptive:
        P, _, _ = fixed_anchor_metric(model,anchor,est,relative_ridge=1e-10)
        L = np.linalg.cholesky(P)
    covariance_error = None if adaptive else float(np.linalg.norm(L@L.T-P)/np.linalg.norm(P))
    metrics = [RMSprop(model.d,alpha=candidate["alpha"],epsilon=candidate["epsilon"]) for _ in range(64)] if adaptive else None
    theta = np.tile(anchor.theta,(64,1)); live = np.ones(64,dtype=bool)
    br = [generator(seed,r,BATCH) for r in range(64)]
    nr = [generator(seed,r,NOISE) for r in range(64)]
    history = {r:deque(maxlen=21) for r in selected}
    vector_events = []
    failures = []; snapshots = []; max_formula_error = 0.; max_metric_error = 0.
    h = candidate["h"]
    original_states = np.load(PILOT/f"states_{name}.npz")
    fields = ["replica","iteration","radius_before","radius_after","gradient_norm","full_gradient_norm",
              "gradient_error_norm","preconditioned_gradient_error_norm","base_norm","prior_norm",
              "correction_norm","remainder_norm","preconditioned_correction_norm","preconditioned_remainder_norm",
              "drift_norm","noise_norm","whitened_drift_norm","whitened_noise_norm",
              "v_before_min","v_before_max","v_after_min","v_after_max","metric_min","metric_max",
              "h_metric_min","h_metric_max","formula_relative_error","metric_relative_error","failed"]
    with (OUT/f"trajectory_{name}.csv").open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for k in range(1,2049):
            for r in range(64):
                if not live[r]: continue
                before = theta[r].copy()
                idx = None if est.method == "ula" else br[r].choice(model.N,32,replace=False)
                z = nr[r].standard_normal(model.d)
                v_before = metrics[r].v.copy() if adaptive else np.zeros(model.d)
                if adaptive:
                    g,raw = est.gradient(before,idx,return_raw_mean=True)
                    D = metrics[r].update(raw)
                    drift = -h*D*g; noise = np.sqrt(2*h*D)*z
                    proposal = before-h*D*g+np.sqrt(2*h*D)*z
                else:
                    g = est.gradient(before,idx)
                    D = None; raw = np.zeros(model.d)
                    drift = -h*(P@g); noise = np.sqrt(2*h)*(L@z)
                    proposal = before-h*(P@g)+np.sqrt(2*h)*(L@z)
                radius = np.linalg.norm((proposal-anchor.theta)@transform)
                failed = not np.isfinite(proposal).all() or not np.isfinite(radius) or radius>100*np.sqrt(model.d)
                if r in selected:
                    direct,base,prior,correction,remainder = independent_components(model,anchor.theta,before,idx,est.method)
                    full = model.full_gradient(before)
                    apply = (lambda x:D*x) if adaptive else (lambda x:P@x)
                    formula_error = float(np.linalg.norm(g-direct)/(1+np.linalg.norm(direct)))
                    metric_error = 0.
                    v_after = metrics[r].v.copy() if adaptive else v_before
                    if adaptive:
                        raw_direct = np.sum(model.X[idx]*(expit(model.X[idx]@before)-model.y[idx])[:,None],axis=0)/32
                        v_direct = candidate["alpha"]*v_before+(1-candidate["alpha"])*raw_direct**2
                        D_direct = 1/(candidate["epsilon"]+np.sqrt(v_direct))
                        metric_error = float(np.linalg.norm(D-D_direct)/(1+np.linalg.norm(D_direct)))
                    max_formula_error = max(max_formula_error,formula_error)
                    max_metric_error = max(max_metric_error,metric_error)
                    n = np.linalg.norm
                    values = [r,k,n((before-anchor.theta)@transform),radius,n(g),n(full),n(g-full),n(apply(g-full)),
                              n(base),n(prior),n(correction),n(remainder),n(apply(correction)),n(apply(remainder)),
                              n(drift),n(noise),n(drift@transform),n(noise@transform),
                              v_before.min(),v_before.max(),v_after.min(),v_after.max(),
                              D.min() if adaptive else None,D.max() if adaptive else None,
                              h*D.min() if adaptive else None,h*D.max() if adaptive else None,
                              formula_error,metric_error,int(failed)]
                    writer.writerow(dict(zip(fields,values)))
                    event = dict(replica=r,iteration=k,theta_before=before,proposal=proposal,g=g,full_gradient=full,
                                 base=base,prior=prior,correction=correction,remainder=remainder,drift=drift,noise=noise,
                                 normal=z,indices=np.array([],dtype=int) if idx is None else idx,
                                 v_before=v_before,v_after=v_after,diagonal=np.zeros(model.d) if D is None else D)
                    history[r].append(event)
                    if adaptive and k<=10: vector_events.append(event)
                    if failed: vector_events.extend(history[r])
                if failed:
                    failures.append(dict(replica=r,iteration=k,reason="nonfinite transition" if not np.isfinite(proposal).all() else "frozen radius limit"))
                    theta[r] = np.nan;live[r] = False
                else: theta[r] = proposal
            if k in [128,512,2048]:
                j=list(original_states["checkpoints"]).index(k)
                snapshots.append(dict(iteration=k,exact_equal=bool(np.array_equal(theta,original_states["endpoints"][j],equal_nan=True))))
    # First ten steps and pre-failure windows; duplicate records removed.
    events = {(e["replica"],e["iteration"]):e for e in vector_events}
    events = list(events.values())
    if events:
        np.savez_compressed(OUT/f"event_vectors_{name}.npz",**{key:np.array([e[key] for e in events]) for key in events[0]})
    final_equal = bool(np.array_equal(theta,original_states["terminal_state"],equal_nan=True))
    adaptive_equal = not adaptive or bool(np.array_equal(np.stack([m.v for m in metrics]),original_states["adaptive_final_v"]))
    result = dict(candidate=candidate,seed=seed,failed_replicas=len(failures),failure_rate=len(failures)/64,
                  failures=failures,failures_exactly_reproduced=failures==original["failures"],checkpoints=snapshots,
                  terminal_exact_equal=final_equal,adaptive_final_state_exact_equal=adaptive_equal,
                  selected_detailed_replicas=selected,max_direct_formula_relative_error=max_formula_error,
                  max_direct_metric_relative_error=max_metric_error,noise_covariance_relative_error=covariance_error,
                  research_seconds=perf_counter()-start,
                  noise_check="fixed: L L.T=P; adaptive: noise coefficient squared=2hD by implemented expression; no Gaussianity test",
                  scope="all 64 replicas replayed; detailed diagnostics only for declared representative IDs; no accuracy scoring or tuning")
    result["exact_replay"] = final_equal and adaptive_equal and result["failures_exactly_reproduced"] and all(x["exact_equal"] for x in snapshots)
    save_json(OUT/f"replay_{name}.json",result)
    print(f"{name}: exact={result['exact_replay']}, failures={len(failures)}, formula_error={max_formula_error:.3g}",flush=True)
    return result


def main():
    start = perf_counter()
    if (OUT/"manifest.json").exists(): raise ValueError("diagnostic output already exists")
    OUT.mkdir(exist_ok=True)
    lock = json.loads((PILOT/"delivery.lock.json").read_text());verify_lock(lock,ROOT)
    config = json.loads((ROOT/"configs/p1_german_frozen_v2.json").read_text())
    chosen = [c for c in config["pilot_candidates"] if
              (c["arm"]=="full_cv2" and c["coefficient"] in [.4,.8]) or
              (c["arm"] in ["full_cv1","full_ula"] and c["coefficient"]==.8) or
              (c["arm"] in ["pcv1","pcv2"] and c["alpha"]==.99 and c["coefficient"]==.256)]
    selected = {}
    for seed in config["seeds"]["pilot"]:
        full = json.loads((PILOT/f"result_full_cv2_c5_seed{seed}.json").read_text())["failures"]
        li = json.loads((PILOT/f"result_pcv2_a0.99_c0.256_seed{seed}.json").read_text())["failures"]
        failed_ids = {f["replica"] for f in full+li}
        selected[seed] = sorted({min(full,key=lambda f:f["iteration"])["replica"],max(full,key=lambda f:f["iteration"])["replica"],
                                 min(li,key=lambda f:f["iteration"])["replica"],next(r for r in range(64) if r not in failed_ids)})
    plan = dict(status="running",purpose="mechanism/implementation diagnosis only",pilot_seeds=config["seeds"]["pilot"],
                candidates=chosen,detailed_replica_selection=selected,selection_rule="earliest/latest full-CV2 failures, earliest Li-CV2 failure, first common survivor; deduplicated",
                selection_heldout_amendment_seeds_consumed=False,steps=2048,replicas=64,
                safeguards="original radius termination only; no clipping, replacement, refresh or initialization changes",
                formula_tolerance=1e-10,source_sha256=sha256(Path(__file__)))
    save_json(OUT/"diagnostic_plan.json",plan)
    save_json(OUT/"manifest.json",plan)
    with threadpool_limits(limits=1):
        model,_=german_credit(ROOT/config["target"]["archive"])
        anchor=find_anchor(model)
        with np.load(ROOT/config["target"]["assets_source"]/"frozen_assets.npz") as assets:
            transform=assets["transform"].copy()
        est=prepare_estimator(model,"cv2",anchor)
        direct_H=sum((w*np.outer(x,x) for w,x in zip(model.weights(anchor.theta),model.X)),np.zeros((model.d,model.d)))
        cache_error=float(np.linalg.norm(est.hessian_hat-direct_H)/np.linalg.norm(direct_H))
        results=[]
        for c in chosen:
            for seed in config["seeds"]["pilot"]:
                results.append(replay(model,anchor,c,seed,selected[seed],transform))
    verify_lock(lock,ROOT)
    save_json(OUT/"manifest.json",dict(plan,status="complete",research_seconds=perf_counter()-start,
        runs=len(results),all_exact_replays=all(r["exact_replay"] for r in results),
        direct_hessian_cache_relative_error=cache_error,
        formula_checks_pass=cache_error<1e-10 and all(r["max_direct_formula_relative_error"]<1e-10 and r["max_direct_metric_relative_error"]<1e-10 for r in results),
        pilot_delivery_lock_unchanged=True))


if __name__=="__main__":main()

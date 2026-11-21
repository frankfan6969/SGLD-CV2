"""Run only the immutable German P1 pilot; never selection/heldout or automatic amendments."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import platform

import numpy as np
from threadpoolctl import threadpool_limits

from freeze_german_p1 import validate_config, array_hash
from p1_bounded_sampler import bounded_replicas
from sgld_cv.anchors import find_anchor
from sgld_cv.data import german_credit
from sgld_cv.diagnostics import gradient_audit
from sgld_cv.estimators import prepare_estimator
from sgld_cv.functionals import FunctionalSpec, functional_rmse
from sgld_cv.preconditioners import fixed_anchor_metric
from sgld_cv.protocol import sha256, verify_lock
from sgld_cv.randomness import generator
from sgld_cv.transport import AccuracyMetric
from sgld_cv.workflow import save_json, save_csv

ROOT = Path(__file__).resolve().parents[1]


def make_schedule(config):
    """Interleave paired seeds within randomized matched-method blocks."""
    blocks = {}
    for candidate in config["pilot_candidates"]:
        key = (candidate["geometry"],candidate.get("alpha"),candidate["coefficient"])
        blocks.setdefault(key,[]).append(candidate)
    rng = np.random.default_rng(config["seeds"]["timing_order"][0])
    keys = list(blocks)
    schedule=[]
    for block_index in rng.permutation(len(keys)):
        block=blocks[keys[block_index]]
        for seed in rng.permutation(config["seeds"]["pilot"]):
            for i in rng.permutation(len(block)):
                schedule.append({"seed":int(seed),"candidate":block[int(i)]})
    if len(schedule)!=config["budgets"]["pilot"]["maximum_candidate_seed_runs"]:
        raise AssertionError("candidate-seed schedule differs from frozen run cap")
    return schedule


def score_checkpoint(saved, f, metric, reference, alternative_reference, target, alternative_target, scales, names):
    if not saved["valid"]:
        return {"iteration":saved["iteration"],"status":"failed_ensemble","sw2":None,"variance_nrmse":None,
                "cost":saved["cost"],"functional_rows":[]}
    estimates,_=f.derive(saved["functional_averages"])
    rows=functional_rmse(estimates,target,scales,names)
    nrmse=np.array([row["normalized_rmse"] for row in rows])
    alternative=np.sqrt(np.mean(((estimates-alternative_target)/scales)**2,axis=0))
    d=len(f.center)
    return {"iteration":saved["iteration"],"status":"ok",
        "sw2":metric.distance(saved["endpoints"],reference),
        "variance_nrmse":float(nrmse[d:2*d].mean()),
        "coordinate_mean_nrmse":float(nrmse[:d].mean()),
        "slow_variance_nrmse":float(nrmse[2*d+1]),
        "alternative_reference_sw2":metric.distance(saved["endpoints"],alternative_reference),
        "alternative_reference_variance_nrmse":float(alternative[d:2*d].mean()),
        "cost":saved["cost"],"functional_rows":rows}


def run(config_path, output):
    start=perf_counter()
    c=json.loads(config_path.read_text())
    freeze=ROOT/"results/p1_german_configuration_v2"
    frozen_lock=json.loads((freeze/"configuration.lock.json").read_text())
    verify_lock(frozen_lock,ROOT)
    if sha256(config_path)!=frozen_lock["files"][str(config_path.relative_to(ROOT))]:
        raise ValueError("pilot configuration differs from frozen input")
    with np.load(freeze/"geometry_and_li_planning.npz") as data:
        geometry={k:data[k] for k in data.files}
    validate_config(c,geometry)
    output.mkdir(parents=True,exist_ok=False)
    stage=c["budgets"]["pilot"]; deadline=start+stage["max_stage_elapsed_seconds"]
    schedule=make_schedule(c)
    source_files=[Path(__file__),ROOT/"scripts/p1_bounded_sampler.py"]
    run_lock={"schema":1,"files":dict(frozen_lock["files"])}
    run_lock["files"].update({str(p.relative_to(ROOT)):sha256(p) for p in source_files})
    save_json(output/"run.lock.json",run_lock)
    save_json(output/"schedule.json",schedule)
    manifest={"status":"running","stage":"pilot","config_sha256":sha256(config_path),
        "platform":platform.platform(),"dtype":"float64","blas_threads":1,
        "pilot_seeds_consumed":c["seeds"]["pilot"],"selection_heldout_seeds_consumed":False,
        "automatic_amendments":False,"planned_runs":len(schedule),"completed_runs":0,
        "scope":"R=64 exploratory screening; no formal R=512 precision/noise-gap claims"}
    save_json(output/"manifest.json",manifest)
    try:
        with threadpool_limits(limits=1):
            model,_=german_credit(ROOT/c["target"]["archive"])
            anchor=find_anchor(model)
            if array_hash(anchor.theta)!=c["target"]["anchor_sha256"]:
                # Different BLAS rounding may be tiny but the frozen target must remain the same.
                np.testing.assert_allclose(anchor.theta,geometry["anchor"],rtol=0,atol=1e-10)
            np.testing.assert_allclose(model.full_hessian(anchor.theta),geometry["H"],atol=1e-9,rtol=0)
            source=ROOT/c["target"]["assets_source"]
            with np.load(source/"frozen_assets.npz") as data:
                assets={k:data[k] for k in data.files}
            f=FunctionalSpec(assets["anchor"],assets["transform"],assets["slow_direction"],model.X[assets["prediction_indices"]])
            metric=AccuracyMetric("standardized_sliced_w2",assets["anchor"],assets["transform"],assets["projections"])
            fmeta=json.loads((source/"functional_reference.json").read_text())
            target,scales=assets["function_targets"],assets["function_scales"]
            alternative_target=target+np.asarray(fmeta["independent_reference_disagreement_normalized"])*scales
            first=assets["reference_calibration"].reshape(-1,model.d)
            second=assets["reference_scoring"].reshape(-1,model.d)
            references={}
            for seed in c["seeds"]["pilot"]:
                reference=second[generator(seed,0,84).choice(len(second),c["accuracy"]["reference_size"],replace=False)]
                alternative=first[generator(seed,0,85).choice(len(first),c["accuracy"]["reference_size"],replace=False)]
                references[seed]=(reference,alternative)
                np.savez_compressed(output/f"reference_seed{seed}.npz",scoring=reference,sensitivity=alternative)
            # Common-location audit is evaluation infrastructure, not sampler work.
            audit_start=perf_counter()
            audit=c["batch_sizes"]
            states=first[generator(c["seeds"]["gradient_audit"][0],0,91).choice(len(first),audit["audit_states"],replace=False)]
            estimators={m:prepare_estimator(model,m,anchor) for m in ("sgld","cv1","cv2","ula")}
            audit_rows=[]
            for b in audit["limited_gradient_MSE_check"]:
                summary,errors=gradient_audit(estimators,states,batch_size=b,batches=audit["fresh_batches_per_state"],
                    seed=c["seeds"]["gradient_audit"][1],metric_matrix=geometry["P_full"])
                audit_rows.extend(summary)
                np.savez_compressed(output/f"gradient_errors_b{b}.npz",states=states,**errors)
            save_csv(output/"gradient_mse.csv",audit_rows)
            manifest["gradient_audit_seconds"]=perf_counter()-audit_start
            manifest["anchor_cost"]=anchor.cost.snapshot()
            completed=[]
            for job_index,job in enumerate(schedule):
                if perf_counter()>=deadline:
                    manifest["status"]="stage_budget_censored";break
                candidate,seed=job["candidate"],job["seed"]
                preparation_start=perf_counter()
                estimator=prepare_estimator(model,candidate["method"],anchor)
                options={}
                if candidate["geometry"]=="full":
                    P,metric_cost,_=fixed_anchor_metric(model,anchor,estimator,relative_ridge=1e-10,absolute_ridge=0.)
                    np.testing.assert_allclose(P,geometry["P_full"],rtol=0,atol=1e-11)
                    options={"matrix":P,"metric_cost":metric_cost}
                elif candidate["geometry"]=="li2016":
                    options={"alpha":candidate["alpha"],"epsilon":candidate["epsilon"]}
                prep_seconds=perf_counter()-preparation_start
                trajectory=bounded_replicas(estimator,initial=anchor.theta,h=candidate["h"],
                    batch_size=candidate["batch_size"] or model.N,replicas=stage["replicas"],
                    steps=stage["iteration_checkpoints"][-1],seed=seed,checkpoints=stage["iteration_checkpoints"],
                    functionals=f,transform=f.transform,max_radius=100*np.sqrt(model.d),
                    max_sampler_seconds=stage["max_candidate_sampler_seconds"],stage_deadline=deadline,**options)
                name=f"{candidate['id']}_seed{seed}"
                saved=trajectory.pop("saved")
                np.savez_compressed(output/f"states_{name}.npz",
                    checkpoints=np.array([x["iteration"] for x in saved],dtype=int),
                    endpoints=np.array([x["endpoints"] for x in saved]),
                    functional_averages=np.array([x["functional_averages"] for x in saved]),
                    terminal_state=trajectory.pop("terminal_state"),
                    terminal_functional_sums=trajectory.pop("terminal_functional_sums"),
                    adaptive_final_v=trajectory.pop("adaptive_final_v"))
                evaluation_start=perf_counter()
                reference,alternative=references[seed]
                scores=[]
                for snapshot in saved:
                    # If stage budget is exhausted, do not launch more expensive evaluation.
                    if perf_counter()>=deadline:
                        scores.append({"iteration":snapshot["iteration"],"status":"evaluation_budget_censored", "cost":snapshot["cost"]})
                    else:
                        scores.append(score_checkpoint(snapshot,f,metric,reference,alternative,target,alternative_target,scales,fmeta["names"]))
                record={"candidate":candidate,"seed":seed,"replicas":stage["replicas"],**trajectory,
                    "preparation_elapsed_seconds":prep_seconds,"checkpoints":scores,
                    "scoring_seconds":perf_counter()-evaluation_start,
                    "interpretation":"descriptive pilot; conditional approximate reference; no formal superiority"}
                save_json(output/f"result_{name}.json",record)
                completed.append({"candidate_id":candidate["id"],"seed":seed,
                    "stop_reason":record["stop_reason"],"failures":len(record["failures"])})
                manifest.update(completed_runs=len(completed),research_elapsed_seconds=perf_counter()-start,
                    last_candidate=candidate["id"],last_seed=seed,
                    completed_with_failures=sum(x["failures"]>0 for x in completed))
                save_json(output/"manifest.json",manifest)
                print(f"{len(completed)}/{len(schedule)} {name}: K={record['completed_iterations']}, failures={len(record['failures'])}, {record['stop_reason']}",flush=True)
            else:
                manifest["status"]="complete"
            manifest["unrun_jobs"]=schedule[len(completed):]
            manifest["selection_heldout_seeds_consumed"]=False
            manifest["research_elapsed_seconds"]=perf_counter()-start
            verify_lock(run_lock,ROOT)
            manifest["run_lock_verified"]=True
    except Exception as exc:
        manifest.update(status="failed",error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save_json(output/"manifest.json",manifest)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/p1_german_frozen_v2.json")
    parser.add_argument("--output",type=Path,default=ROOT/"results/p1_german_pilot_v1")
    args=parser.parse_args()
    run(args.config.resolve(),args.output.resolve())


if __name__=="__main__":main()

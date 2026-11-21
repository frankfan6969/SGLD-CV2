"""Execute only the frozen heldout union; no selection or reoptimization."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
from freeze_german_p1 import validate_config,array_hash
from p1_bounded_sampler import bounded_replicas
from run_german_p1_pilot import score_checkpoint
from sgld_cv.anchors import find_anchor
from sgld_cv.data import german_credit
from sgld_cv.estimators import prepare_estimator
from sgld_cv.functionals import FunctionalSpec
from sgld_cv.preconditioners import fixed_anchor_metric
from sgld_cv.protocol import sha256,verify_lock
from sgld_cv.randomness import generator
from sgld_cv.transport import AccuracyMetric
from sgld_cv.workflow import save_json

ROOT=Path(__file__).resolve().parents[1]


def schedule(e):
    blocks={}
    for c in e["candidates"]:blocks.setdefault((c["geometry"],c["alpha"],c["coefficient"]),[]).append(c)
    rng=np.random.default_rng(e["timing_order_seed"]);keys=list(blocks);jobs=[]
    for j in rng.permutation(len(keys)):
        block=blocks[keys[j]]
        for seed in rng.permutation(e["seeds"]):
            for i in rng.permutation(len(block)):jobs.append(dict(candidate=block[i],seed=int(seed)))
    return jobs


def validate_execution(e,c):
    if e["stage"]!="heldout" or e["seeds"]!=c["seeds"]["heldout"]:
        raise ValueError("heldout role/seeds mismatch")
    if e["budget"]!=c["budgets"]["heldout"]:raise ValueError("heldout budget changed")
    selection=json.loads((ROOT/"configs/p1_german_selection_v1.json").read_text())
    choices=json.loads((ROOT/"results/p1_german_selection_v1/final_choices.json").read_text())
    ids={q["choice"]["candidate_id"] for q in choices["choices"] if q["choice"]}
    ids|={q["candidate_id"] for q in choices["matched_cells"]}
    expected=[q for q in selection["candidates"] if q["id"] in ids]
    if e["candidates"]!=expected:raise ValueError("heldout candidates differ from locked selection union")
    if len(expected)*len(e["seeds"])>c["budgets"]["heldout"]["maximum_candidate_seed_runs"]:
        raise ValueError("heldout run cap exceeded")
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True);args=parser.parse_args()
    config_path=args.config.resolve();e=json.loads(config_path.read_text());c=json.loads((ROOT/e["parent_config"]).read_text())
    lock=json.loads((ROOT/e["execution_lock"]).read_text());verify_lock(lock,ROOT)
    validate_execution(e,c)
    with np.load(ROOT/"results/p1_german_configuration_v2/geometry_and_li_planning.npz") as data:geometry={k:data[k] for k in data.files}
    validate_config(c,geometry)
    output=ROOT/e["output"];output.mkdir(exist_ok=False,parents=True)
    start=perf_counter();stage=e["budget"];deadline=start+stage["max_stage_elapsed_seconds"]
    jobs=schedule(e);save_json(output/"schedule.json",jobs);save_json(output/"run.lock.json",lock)
    manifest=dict(status="running",stage=e["stage"],planned_runs=len(jobs),completed_runs=0,
        stage_seeds=e["seeds"],heldout_seeds_consumed=True,config_sha256=sha256(config_path),
        research_elapsed_seconds=0.,blas_threads=1,dtype="float64",scope="heldout evaluation of previously locked configurations; no retuning")
    save_json(output/"manifest.json",manifest)
    try:
        with threadpool_limits(limits=1):
            model,_=german_credit(ROOT/c["target"]["archive"]);anchor=find_anchor(model)
            np.testing.assert_allclose(anchor.theta,geometry["anchor"],atol=1e-10,rtol=0)
            with np.load(ROOT/c["target"]["assets_source"]/"frozen_assets.npz") as data:a={k:data[k] for k in data.files}
            f=FunctionalSpec(a["anchor"],a["transform"],a["slow_direction"],model.X[a["prediction_indices"]])
            metric=AccuracyMetric("standardized_sliced_w2",a["anchor"],a["transform"],a["projections"])
            meta=json.loads((ROOT/c["target"]["assets_source"]/"functional_reference.json").read_text())
            target,scales=a["function_targets"],a["function_scales"]
            alternative_target=target+np.asarray(meta["independent_reference_disagreement_normalized"])*scales
            pools=[a["reference_scoring"].reshape(-1,model.d),a["reference_calibration"].reshape(-1,model.d)]
            refs={}
            for seed in e["seeds"]:
                refs[seed]=tuple(pool[generator(seed,0,84+j).choice(len(pool),c["accuracy"]["reference_size"],replace=False)] for j,pool in enumerate(pools))
                np.savez_compressed(output/f"reference_seed{seed}.npz",scoring=refs[seed][0],sensitivity=refs[seed][1])
            manifest["anchor_cost"]=anchor.cost.snapshot()
            completed=[]
            for job in jobs:
                if perf_counter()>=deadline:manifest["status"]="stage_budget_censored";break
                candidate,seed=job["candidate"],job["seed"];t=perf_counter()
                est=prepare_estimator(model,candidate["method"],anchor);options={}
                if candidate["geometry"]=="full":
                    P,cost,_=fixed_anchor_metric(model,anchor,est,relative_ridge=1e-10)
                    np.testing.assert_allclose(P,geometry["P_full"],atol=1e-11,rtol=0)
                    options=dict(matrix=P,metric_cost=cost)
                elif candidate["geometry"]=="li2016":options=dict(alpha=candidate["alpha"],epsilon=candidate["epsilon"])
                prep=perf_counter()-t
                run=bounded_replicas(est,initial=anchor.theta,h=candidate["h"],batch_size=candidate["batch_size"] or model.N,
                    replicas=stage["replicas"],steps=stage["iteration_checkpoints"][-1],seed=seed,
                    checkpoints=stage["iteration_checkpoints"],functionals=f,transform=f.transform,max_radius=100*np.sqrt(model.d),
                    max_sampler_seconds=stage["max_candidate_sampler_seconds"],stage_deadline=deadline,**options)
                saved=run.pop("saved");name=f"{candidate['id']}_seed{seed}"
                np.savez_compressed(output/f"states_{name}.npz",checkpoints=np.array([s["iteration"] for s in saved],dtype=int),
                    endpoints=np.array([s["endpoints"] for s in saved]),functional_averages=np.array([s["functional_averages"] for s in saved]),
                    terminal_state=run.pop("terminal_state"),terminal_functional_sums=run.pop("terminal_functional_sums"),adaptive_final_v=run.pop("adaptive_final_v"))
                t=perf_counter();scores=[]
                for s in saved:
                    scores.append(dict(iteration=s["iteration"],status="evaluation_budget_censored",cost=s["cost"]) if perf_counter()>=deadline else
                        score_checkpoint(s,f,metric,*refs[seed],target,alternative_target,scales,meta["names"]))
                record=dict(candidate=candidate,seed=seed,replicas=stage["replicas"],**run,checkpoints=scores,
                    preparation_elapsed_seconds=prep,scoring_seconds=perf_counter()-t)
                save_json(output/f"result_{name}.json",record);completed.append(name)
                manifest.update(completed_runs=len(completed),research_elapsed_seconds=perf_counter()-start,last_run=name)
                save_json(output/"manifest.json",manifest)
                print(f"{len(completed)}/{len(jobs)} {name}: K={run['completed_iterations']}, failures={len(run['failures'])}, {run['stop_reason']}",flush=True)
            else:manifest["status"]="complete"
            manifest["unrun_jobs"]=jobs[len(completed):];verify_lock(lock,ROOT);manifest["run_lock_verified"]=True
    except Exception as exc:
        manifest.update(status="failed",error=f"{type(exc).__name__}: {exc}");raise
    finally:
        manifest["research_elapsed_seconds"]=perf_counter()-start;save_json(output/"manifest.json",manifest)


if __name__=="__main__":main()

"""Complete predeclared Selection diagnostics/timing without changing choices."""
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
from sgld_cv.anchors import find_anchor
from sgld_cv.data import german_credit
from sgld_cv.diagnostics import chain_diagnostics
from sgld_cv.estimators import prepare_estimator
from sgld_cv.preconditioners import fixed_anchor_metric
from sgld_cv.protocol import sha256,verify_lock
from sgld_cv.randomness import generator,INITIALIZATION
from sgld_cv.samplers import run_chains,run_with_theory_step
from sgld_cv.workflow import save_json,save_csv

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"results/p1_german_selection_v1"
OUT=ROOT/"results/p1_german_selection_diagnostics_v1"


def main():
    started=perf_counter()
    locked=json.loads((SOURCE/"final_choices.lock.json").read_text());verify_lock(locked,ROOT)
    e=json.loads((ROOT/"configs/p1_german_selection_v1.json").read_text())
    c=json.loads((ROOT/e["parent_config"]).read_text());choices=json.loads((SOURCE/"final_choices.json").read_text())
    sampled=json.loads((SOURCE/"manifest.json").read_text())
    remaining=c["budgets"]["selection"]["max_stage_elapsed_seconds"]-sampled["research_elapsed_seconds"]
    if remaining<=0:raise ValueError("selection research budget exhausted")
    pairs=sorted({(x["choice"]["candidate_id"],x["choice"]["iteration"]) for x in choices["choices"] if x["choice"]})
    finalist_ids={i for i,k in pairs}|{x["candidate_id"] for x in choices["matched_cells"]}
    finalists=[x for x in e["candidates"] if x["id"] in finalist_ids]
    OUT.mkdir(exist_ok=False)
    plan=dict(stage="selection supplemental diagnostics",finalists=finalists,single_chain_pairs=pairs,
        diagnostic_seeds=[s+100000 for s in e["seeds"]],initialization="anchor + 2*N(0,H^-1), shared per diagnostic seed; INITIALIZATION stream",
        warmup=1000,draws=4000,chains=4,single_chain_seeds=c["seeds"]["single_chain_timing"],
        remaining_stage_research_seconds=remaining,single_chain_cap_seconds=1800,
        failure_rule="nonfinite or original whitened radius>100sqrt(d), no filtering",
        original_choices_unchanged=True,heldout_started=False)
    save_json(OUT/"protocol.json",plan)
    lock=dict(schema=1,files=dict(locked["files"]))
    for p in [Path(__file__),OUT/"protocol.json"]:lock["files"][str(p.relative_to(ROOT))]=sha256(p)
    save_json(OUT/"execution.lock.json",lock)
    deadline=started+remaining;diagnostics=[];timings=[];unrun=[]
    with threadpool_limits(limits=1):
        model,_=german_credit(ROOT/c["target"]["archive"]);anchor=find_anchor(model)
        H=model.full_hessian(anchor.theta);inverse=np.linalg.solve(H,np.eye(model.d));L=np.linalg.cholesky((inverse+inverse.T)/2)
        with np.load(ROOT/c["target"]["assets_source"]/"frozen_assets.npz") as a:T=a["transform"].copy()
        def prepare(candidate):
            est=prepare_estimator(model,candidate["method"],anchor)
            options=dict(failure_center=anchor.theta,failure_transform=T,max_standardized_radius=100*np.sqrt(model.d))
            if candidate["geometry"]=="full":
                P,cost,_=fixed_anchor_metric(model,anchor,est,relative_ridge=1e-10)
                options.update(preconditioner="frozen_full",frozen_matrix=P,metric_cost=cost)
            elif candidate["geometry"]=="li2016":
                options.update(preconditioner="rmsprop",alpha=candidate["alpha"],epsilon=candidate["epsilon"])
            return est,options
        for candidate in finalists:
            for seed in plan["diagnostic_seeds"]:
                if perf_counter()>=deadline:
                    unrun.append(dict(kind="diagnostic",candidate_id=candidate["id"],seed=seed));continue
                initial=anchor.theta+2*generator(seed,0,INITIALIZATION).normal(size=(4,model.d))@L.T
                est,options=prepare(candidate)
                samples,run=run_chains(est,initial=initial,step_size=2*candidate["h"],batch_size=candidate["batch_size"] or model.N,
                    warmup=1000,draws=4000,chains=4,seed=seed,**options)
                diag=chain_diagnostics(samples,cost=run.costs[-1],adaptive=candidate["geometry"]=="li2016")
                row=dict(candidate_id=candidate["id"],seed=seed,diagnostics=diag,cost=run.costs[-1],failures=run.failures)
                diagnostics.append(row);save_json(OUT/f"diagnostic_{candidate['id']}_seed{seed}.json",row)
                np.savez_compressed(OUT/f"diagnostic_{candidate['id']}_seed{seed}.npz",samples=samples,initial=initial)
                print(f"diagnostic {candidate['id']} seed{seed}: reliable={diag['reliable']}",flush=True)
        timing_start=perf_counter()
        for candidate_id,k in pairs:
            candidate=next(x for x in finalists if x["id"]==candidate_id)
            for seed in plan["single_chain_seeds"]:
                if perf_counter()>=deadline or perf_counter()-timing_start>=1800:
                    unrun.append(dict(kind="single_chain",candidate_id=candidate_id,iteration=k,seed=seed));continue
                est,options=prepare(candidate)
                run=run_with_theory_step(est,h=candidate["h"],initial=anchor.theta,batch_size=candidate["batch_size"] or model.N,
                    steps=k,replicas=1,seed=seed,checkpoints=[k],cost_snapshots=False,**options)
                timings.append(dict(candidate_id=candidate_id,iteration=k,seed=seed,total_seconds=run.costs[-1]["total_seconds"],
                    failures=len(run.failures),cost=run.costs[-1]))
        save_json(OUT/"single_chain_timings.json",timings)
    summary=[]
    for candidate in finalists:
        rows=[r for r in diagnostics if r["candidate_id"]==candidate["id"]]
        fixed=candidate["geometry"]!="li2016"
        summary.append(dict(candidate_id=candidate["id"],runs=len(rows),failed_replicas=sum(len(r["failures"]) for r in rows),
            fixed_metric=fixed,mixing_verified=fixed and len(rows)==4 and all(r["diagnostics"]["reliable"] for r in rows),
            max_rhat=max((r["diagnostics"].get("max_rhat",float("inf")) for r in rows),default=None),
            min_bulk_ess=min((r["diagnostics"].get("min_bulk_ess",0) for r in rows),default=None),
            min_tail_ess=min((r["diagnostics"].get("min_tail_ess",0) for r in rows),default=None),
            interpretation="adaptive ESS exploratory" if not fixed else "gate only, not posterior accuracy guarantee"))
    save_csv(OUT/"mixing_summary.csv",summary)
    timing_summary=[]
    for candidate,k in pairs:
        rows=[r for r in timings if r["candidate_id"]==candidate and r["iteration"]==k]
        timing_summary.append(dict(candidate_id=candidate,iteration=k,repeats=len(rows),failed=sum(r["failures"] for r in rows),
            median_seconds=float(np.median([r["total_seconds"] for r in rows])) if rows else None,
            min_seconds=min((r["total_seconds"] for r in rows),default=None),max_seconds=max((r["total_seconds"] for r in rows),default=None)))
    save_csv(OUT/"single_chain_summary.csv",timing_summary)
    verify_lock(lock,ROOT)
    elapsed=perf_counter()-started
    save_json(OUT/"manifest.json",dict(status="complete" if not unrun else "budget_censored",research_seconds=elapsed,
        total_selection_research_seconds=sampled["research_elapsed_seconds"]+elapsed,diagnostic_runs=len(diagnostics),
        timing_runs=len(timings),unrun=unrun,heldout_started=False,final_choices_unchanged=True))
    paths=[p for p in OUT.iterdir() if p.is_file()]
    save_json(OUT/"delivery.lock.json",dict(schema=1,files={str(p.relative_to(ROOT)):sha256(p) for p in paths}))


if __name__=="__main__":main()

"""Frozen CPU-config replication on A100, with independent GPU RNG and costs."""
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from functools import lru_cache
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from batched_sampler import prepare,initial_state,advance,chunk_kernel,functional_chunk,ready

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(ROOT/"scripts"))
from sgld_cv.anchors import find_anchor
from sgld_cv.models import LogisticRegression
from sgld_cv.functionals import FunctionalSpec
from sgld_cv.transport import AccuracyMetric
from sgld_cv.randomness import generator
from sgld_cv.protocol import sha256,verify_lock
from sgld_cv.workflow import save_json
from run_german_p1_pilot import score_checkpoint
from run_german_p1_stage import schedule
from threadpoolctl import threadpool_limits


@lru_cache(maxsize=32)
def instrumented_kernel(method,geometry,batch,alpha,epsilon):
    def run(p,state,reasons,seed,h):
        R,d=state[0].shape;N=p["X"].shape[0]
        keys=jax.vmap(lambda r:jax.random.fold_in(jax.random.PRNGKey(seed),r))(jnp.arange(R,dtype=jnp.uint32))
        bk=jax.vmap(lambda k:jax.random.fold_in(k,1))(keys);nk=jax.vmap(lambda k:jax.random.fold_in(k,2))(keys)
        def body(carry,_):
            old,reason=carry;t=old[4]
            z=jax.vmap(lambda k:jax.random.normal(jax.random.fold_in(k,t),(d,),dtype=jnp.float64))(nk)
            idx=jnp.zeros((R,batch),dtype=jnp.int32) if method=="ula" else jax.vmap(lambda k:jax.random.choice(jax.random.fold_in(k,t),N,(batch,),replace=False))(bk)
            new,public,parts=advance(p,old,idx,z,h,method,geometry,"logistic",alpha,epsilon)
            proposal=old[0]+parts[4]+parts[5]
            code=jnp.where(jnp.all(jnp.isfinite(proposal),axis=1),2,1)
            reason=jnp.where(old[2]&~new[2],code,reason)
            return (new,reason),public
        (state,reasons),states=lax.scan(body,(state,reasons),None,length=128)
        return state,reasons,states
    return jax.jit(run)


def main():
    start=perf_counter();protocol=json.loads((ROOT/"gpu_replication/FULL_PROTOCOL.json").read_text())
    verify_lock(json.loads((ROOT/"gpu_replication/full.lock.json").read_text()),ROOT)
    gate=json.loads((ROOT/"results/gpu_benchmark_a100_v1/REPORT.json").read_text())
    if not gate["full_replication_performance_gate"]:raise RuntimeError("performance gate not passed")
    device=jax.devices("gpu")[0]
    if "A100" not in device.device_kind or jax.__version__!=protocol["jax_version"]:raise RuntimeError("GPU/runtime version differs from freeze")
    c=json.loads((ROOT/"configs/p1_german_frozen_v2.json").read_text());e=json.loads((ROOT/"configs/p1_german_heldout_v1.json").read_text())
    output=ROOT/"results/p1_german_gpu_replication_v1";output.mkdir(exist_ok=False,parents=True)
    run_lock=json.loads((ROOT/"gpu_replication/full.lock.json").read_text());save_json(output/"run.lock.json",run_lock)
    jobs=schedule(e);save_json(output/"schedule.json",jobs);deadline=start+43200
    manifest=dict(status="running",stage="GPU frozen-config replication",planned_runs=len(jobs),completed_runs=0,
        heldout_seeds_consumed=True,backend="JAX CUDA",jax=jax.__version__,numpy=np.__version__,device=device.device_kind,
        x64=jax.config.x64_enabled,research_elapsed_seconds=0.,GPU_specific_tuning=False,CPU_results_unchanged=True,
        cost_primary="cold first-use attributed sampler compile+anchor+actual transfer/cache/metric+sampling; warm costs also saved. Functional evaluation excluded.")
    save_json(output/"manifest.json",manifest)
    with threadpool_limits(limits=1):
        with np.load(ROOT/"gpu_replication/german_target.npz") as z:data={k:z[k] for k in z.files}
        with np.load(ROOT/c["target"]["assets_source"]/"frozen_assets.npz") as z:a={k:z[k] for k in z.files}
        model=LogisticRegression(data["X"],data["y"]);anchor=find_anchor(model)
        np.testing.assert_allclose(anchor.theta,a["anchor"],atol=1e-9,rtol=0)
        anchor_seconds=anchor.cost.snapshot()["total_seconds"]
        f=FunctionalSpec(a["anchor"],a["transform"],a["slow_direction"],data["prediction_rows"])
        metric=AccuracyMetric("standardized_sliced_w2",a["anchor"],a["transform"],a["projections"])
        meta=json.loads((ROOT/c["target"]["assets_source"]/"functional_reference.json").read_text())
        target,scales=a["function_targets"],a["function_scales"]
        alt_target=target+np.asarray(meta["independent_reference_disagreement_normalized"])*scales
        pools=[a["reference_scoring"].reshape(-1,61),a["reference_calibration"].reshape(-1,61)];refs={}
        for seed in e["seeds"]:
            refs[seed]=tuple(pool[generator(seed,0,84+j).choice(len(pool),1024,replace=False)] for j,pool in enumerate(pools))
            np.savez_compressed(output/f"reference_seed{seed}.npz",scoring=refs[seed][0],sensitivity=refs[seed][1])
        compiled={};compilation={};physical_compile=0.;fexe=None
        for index,job in enumerate(jobs):
            if perf_counter()>=deadline:manifest["status"]="stage_budget_censored";break
            candidate,seed=job["candidate"],job["seed"];method=candidate["method"];geometry=candidate["geometry"]
            alpha=candidate["alpha"] or .99;epsilon=candidate["epsilon"] or 1e-5
            p,setup=prepare(data["X"],data["y"],anchor.theta,1.,"logistic",method,geometry,a["transform"],device=device)
            setup["operations"]=dict(anchor_residual_evaluations=1000,
                observation_gradient_evaluations=1000 if method in ["cv1","cv2"] else 0,
                observation_hessian_evaluations=1000 if method=="cv2" or geometry=="full" else 0,
                dense_hessian_assemblies=int(method=="cv2" or geometry=="full"),
                dense_matrix_solves=int(geometry=="full"),cholesky_factorizations=2*int(geometry=="full"),
                cached_hessian_reuses=int(method=="cv2" and geometry=="full"))
            setup["device_parameter_bytes"]=sum(x.nbytes for x in p.values())
            initialization_start=perf_counter();state=ready(initial_state(p,512));reasons=ready(jnp.zeros(512,dtype=jnp.int32))
            setup["state_initialization_seconds"]=perf_counter()-initialization_start
            setup["total_seconds"]+=setup["state_initialization_seconds"]
            key=(method,geometry,32,alpha,epsilon);kernel=instrumented_kernel(*key)
            if key not in compiled:
                t=perf_counter();compiled[key]=kernel.lower(p,state,reasons,seed,candidate["h"]).compile()
                compilation[key]=perf_counter()-t;physical_compile+=compilation[key]
                # Check instrumentation against the validated original random-stream kernel.
                diagnostic_start=perf_counter();short=initial_state(p,4)
                aa,rr,xx=ready(kernel(p,short,jnp.zeros(4,dtype=jnp.int32),144400,candidate["h"]))
                bb,yy=ready(chunk_kernel(method,geometry,"logistic",32,128,alpha,epsilon)(p,short,144400,candidate["h"]))
                np.testing.assert_allclose(xx,yy,rtol=1e-9,atol=1e-8,equal_nan=True)
                np.testing.assert_array_equal(aa[3],bb[3])
                manifest["instrumentation_verification_seconds"]=manifest.get("instrumentation_verification_seconds",0.)+perf_counter()-diagnostic_start
            sampler=functional_seconds=0.;saved=[];stop="complete";sums=jnp.zeros((512,132),dtype=jnp.float64)
            fd=(p["anchor"],p["transform"],jnp.asarray(a["slow_direction"]),jnp.asarray(data["prediction_rows"]))
            def snapshot():
                k=int(state[4])
                seconds=dict(anchor_optimization=anchor_seconds,gpu_transfer=setup["transfer_seconds"],gpu_cache_metric=setup["cache_metric_seconds"],
                    gpu_state_initialization=setup["state_initialization_seconds"],sampler_compilation_attributed=compilation[key],sampling=sampler,burn_in=0.)
                operations=dict(device_lane_iterations=512*k,logical_live_transitions=int(state[5]),transitions=512*k,
                    observation_gradient_evaluations=(1000 if method=="ula" else 32)*512*k,
                    normal_variates=61*512*k,prior_gradient_evaluations=512*k,
                    dense_hessian_vector_products=512*k if method=="cv2" else 0,
                    metric_updates=512*k if geometry=="li2016" else 0)
                for source in [anchor.cost.snapshot()["operations"],setup["operations"]]:
                    for key2,value in source.items():operations[key2]=operations.get(key2,0)+value
                return dict(seconds=seconds,total_seconds=sum(seconds.values()),operations=operations,
                    warm_total_seconds=sum(seconds.values())-compilation[key],scope="device-padded conceptual work counters; anchor and setup included; no gradient-equivalent conversion")
            for block in range(64):
                if perf_counter()>=deadline:stop="stage_budget_censored";break
                if sampler+setup["total_seconds"]+anchor_seconds+compilation[key]>=1800:stop="candidate_budget_censored";break
                t=perf_counter();state,reasons,states=ready(compiled[key](p,state,reasons,seed,candidate["h"]))
                sampler+=perf_counter()-t;k=int(state[4])
                if fexe is None:
                    t=perf_counter();fexe=functional_chunk.lower(states,*fd).compile();manifest["functional_compilation_seconds"]=perf_counter()-t
                t=perf_counter();sums=sums+fexe(states,*fd);ready(sums);functional_seconds+=perf_counter()-t
                if k in [128,512,2048,8192]:
                    cost=snapshot()
                    saved.append(dict(iteration=k,endpoints=np.asarray(states[-1]),functional_averages=np.asarray(sums)/k,cost=cost,valid=bool(np.asarray(state[2]).all())))
            failed_at=np.asarray(state[3]);codes=np.asarray(reasons)
            failures=[dict(replica=int(r),iteration=int(failed_at[r]),reason="nonfinite transition" if codes[r]==1 else "frozen radius limit") for r in np.flatnonzero(failed_at)]
            scores=[];t=perf_counter()
            for snap in saved:
                scores.append(dict(iteration=snap["iteration"],status="evaluation_budget_censored",cost=snap["cost"]) if perf_counter()>=deadline else
                    score_checkpoint(snap,f,metric,*refs[seed],target,alt_target,scales,meta["names"]))
            scoring_seconds=perf_counter()-t;name=f"{candidate['id']}_seed{seed}"
            np.savez_compressed(output/f"states_{name}.npz",checkpoints=np.array([s["iteration"] for s in saved],dtype=int),
                endpoints=np.array([s["endpoints"] for s in saved]),functional_averages=np.array([s["functional_averages"] for s in saved]),
                terminal_state=np.where(np.asarray(state[2])[:,None],np.asarray(state[0]),np.nan),terminal_functional_sums=np.asarray(sums),adaptive_final_v=np.asarray(state[1]))
            save_json(output/f"result_{name}.json",dict(candidate=candidate,seed=seed,replicas=512,checkpoints=scores,failures=failures,
                stop_reason=stop,completed_iterations=int(state[4]),functional_evaluation_seconds=functional_seconds,scoring_seconds=scoring_seconds,
                terminal_cost=snapshot(),setup=setup,sampler_compilation_attributed=compilation[key],sampler_seconds=sampler,
                rng="JAX independent replica/purpose/global-iteration fold_in; distinct from NumPy RNG",precision="float64"))
            manifest.update(completed_runs=index+1,research_elapsed_seconds=perf_counter()-start,last_run=name,physical_sampler_compilation_seconds=physical_compile)
            save_json(output/"manifest.json",manifest)
            print(f"{index+1}/{len(jobs)} {name}: K={int(state[4])}, failed={len(failures)}, sampling={sampler:.3f}s",flush=True)
        else:manifest["status"]="complete"
    manifest["unrun_jobs"]=jobs[manifest["completed_runs"]:];manifest["research_elapsed_seconds"]=perf_counter()-start
    verify_lock(run_lock,ROOT);manifest["run_lock_verified"]=True;save_json(output/"manifest.json",manifest)


if __name__=="__main__":main()

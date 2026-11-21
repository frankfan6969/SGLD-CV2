"""Representative A100 benchmark, gated by GPU numerical validation."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
import numpy as np
import jax
from batched_sampler import prepare,initial_state,chunk_kernel,functional_chunk,ready

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(ROOT/"scripts"))
from sgld_cv.models import LogisticRegression
from sgld_cv.anchors import find_anchor
from sgld_cv.functionals import FunctionalSpec
from sgld_cv.estimators import prepare_estimator
from sgld_cv.preconditioners import fixed_anchor_metric
from p1_bounded_sampler import bounded_replicas
from threadpoolctl import threadpool_limits


def run(output):
    output=Path(output);output.mkdir(exist_ok=False,parents=True)
    gate=json.loads((ROOT/"results/gpu_validation_a100_v1/REPORT.json").read_text())
    if gate["status"]!="passed" or gate["platform"]!="gpu" or "A100" not in gate["device_kind"]:raise RuntimeError("GPU validation gate failed")
    gpu=jax.devices("gpu")[0];cpu=jax.devices("cpu")[0]
    with np.load(ROOT/"gpu_replication/german_target.npz") as z:a={k:z[k] for k in z.files}
    model=LogisticRegression(a["X"],a["y"]);records=[]
    with threadpool_limits(limits=1):
        anchor=find_anchor(model)
        np.testing.assert_allclose(anchor.theta,a["anchor"],atol=1e-9,rtol=0)
        f=FunctionalSpec(a["anchor"],a["transform"],a["slow_direction"],a["prediction_rows"])
        for method in ["cv1","cv2","ula"]:
            for device in [cpu,gpu]:
                with jax.default_device(device):
                    p,setup=prepare(a["X"],a["y"],anchor.theta,1.,"logistic",method,"full",a["transform"],device=device)
                    state=initial_state(p,512);ready(state)
                    kernel=chunk_kernel(method,"full","logistic",32,128)
                    t=perf_counter();executable=kernel.lower(p,state,144300,.1).compile();compile_seconds=perf_counter()-t
                    dummy=jax.device_put(np.zeros((128,512,61),dtype=np.float64),device)
                    fd=(p["anchor"],p["transform"],jax.device_put(a["slow_direction"],device),jax.device_put(a["prediction_rows"],device))
                    t=perf_counter();fexe=functional_chunk.lower(dummy,*fd).compile();functional_compile=perf_counter()-t
                    sampler_seconds=functional_seconds=0.;sums=None
                    for _ in range(16):
                        t=perf_counter();state,states=ready(executable(p,state,144300,.1));sampler_seconds+=perf_counter()-t
                        t=perf_counter();values=ready(fexe(states,*fd));sums=values if sums is None else sums+values;ready(sums)
                        functional_seconds+=perf_counter()-t
                    t=perf_counter();endpoints=np.asarray(states[-1]);averages=np.asarray(sums)/2048;host_seconds=perf_counter()-t
                    rec=dict(method=method,backend=f"jax_{device.platform}",device_kind=device.device_kind,R=512,T=2048,batch=32,h=.1,
                        seed=144300,sampler_seconds=sampler_seconds,functional_seconds=functional_seconds,compile_seconds=compile_seconds,
                        functional_compile_seconds=functional_compile,host_output_seconds=host_seconds,setup=setup,
                        anchor_seconds=anchor.cost.snapshot()["total_seconds"],
                        warm_setup_inclusive_seconds=setup["total_seconds"]+anchor.cost.snapshot()["total_seconds"]+sampler_seconds,
                        cold_end_to_end_seconds=setup["total_seconds"]+anchor.cost.snapshot()["total_seconds"]+sampler_seconds+functional_seconds+compile_seconds+functional_compile+host_seconds,
                        failed=int(np.count_nonzero(np.asarray(state[3]))),logical_transitions=int(state[5]),device_lane_iterations=512*2048,
                        accounting="sampling excludes separately timed online functionals; includes GPU RNG and padded failed-lane work")
                    records.append(rec)
                    np.savez_compressed(output/f"states_{method}_{device.platform}.npz",endpoints=endpoints,functional_averages=averages,failed_at=np.asarray(state[3]))
                    (output/"partial.json").write_text(json.dumps(records,indent=2)+"\n")
                    print(method,rec["backend"],"sampling",sampler_seconds,"compile",compile_seconds,flush=True)
            # Same Colab host, original sequential NumPy implementation, separate RNG family.
            est=prepare_estimator(model,method,anchor);P,cost,_=fixed_anchor_metric(model,anchor,est,relative_ridge=1e-10)
            t=perf_counter()
            original=bounded_replicas(est,initial=anchor.theta,h=.1,batch_size=32 if method!="ula" else model.N,
                replicas=512,steps=2048,seed=144300,checkpoints=[2048],functionals=f,transform=f.transform,
                max_radius=100*np.sqrt(61),matrix=P,metric_cost=cost,max_sampler_seconds=600)
            records.append(dict(method=method,backend="numpy_sequential_same_host",R=512,T=2048,
                total_seconds=original["terminal_cost"]["total_seconds"],functional_seconds=original["functional_evaluation_seconds"],
                research_seconds=perf_counter()-t,failed=len(original["failures"]),completed_iterations=original["completed_iterations"]))
    speedups={}
    for method in ["cv1","cv2","ula"]:
        gs=next(r for r in records if r["method"]==method and r["backend"]=="jax_gpu")
        ns=next(r for r in records if r["method"]==method and r["backend"]=="numpy_sequential_same_host")
        cs=next(r for r in records if r["method"]==method and r["backend"]=="jax_cpu")
        speedups[method]=dict(vs_sequential_numpy=ns["total_seconds"]/gs["warm_setup_inclusive_seconds"],
            vs_batched_jit_cpu=cs["warm_setup_inclusive_seconds"]/gs["warm_setup_inclusive_seconds"])
    result=dict(status="complete",jax=jax.__version__,x64=jax.config.x64_enabled,records=records,speedups=speedups,
        full_replication_performance_gate=all(x["vs_sequential_numpy"]>=2 for x in speedups.values()),
        full_replication_started=False,scope="engineering backend benchmark, not statistical efficiency evidence; one representative run per method/backend; CPU/GPU RNG sequences differ")
    (output/"REPORT.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="records"}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);run(p.parse_args().output)

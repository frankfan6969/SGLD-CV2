"""Common-input NumPy/JAX checks; run on GPU again before performance gate."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import jax
import jax.numpy as jnp
from batched_sampler import prepare,initial_state,fixed_input_kernel,chunk_kernel,functional_chunk,ready

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from sgld_cv.models import LogisticRegression,GaussianRegression
from sgld_cv.anchors import supplied_anchor
from sgld_cv.estimators import prepare_estimator
from sgld_cv.functionals import FunctionalSpec
from sgld_cv.preconditioners import RMSprop


def run(output,require_gpu=False):
    device=jax.devices()[0]
    if require_gpu and (device.platform!="gpu" or "A100" not in device.device_kind):raise RuntimeError(f"A100 required, found {device}")
    output=Path(output);output.mkdir(exist_ok=False,parents=True)
    rng=np.random.default_rng(144000);rows=[]
    for kind in ["gaussian","logistic"]:
        X=rng.normal(size=(37,5));y=rng.normal(size=37) if kind=="gaussian" else rng.integers(0,2,37)
        model=GaussianRegression(X,y) if kind=="gaussian" else LogisticRegression(X,y)
        anchor=supplied_anchor(rng.normal(scale=.1,size=5),provenance="common-input validation")
        f=FunctionalSpec.from_geometry(anchor.theta,model.full_hessian(anchor.theta))
        for geometry in ["identity","full","li2016"]:
            for method in ["sgld","cv1","cv2","ula"]:
                p,_=prepare(X,y,anchor.theta,1.,kind,method,geometry,f.transform,device=device)
                est=prepare_estimator(model,method,anchor)
                kernel=fixed_input_kernel(method,geometry,kind)
                state=initial_state(p,4);theta=np.tile(anchor.theta,(4,1));metrics=[RMSprop(5) for _ in range(4)]
                maxg=maxstep=maxv=0.;saved=[]
                for k in range(12):
                    idx=np.stack([rng.choice(37,4,replace=False) for _ in range(4)]);z=rng.normal(size=(4,5));h=.0001
                    next_state,public,parts=ready(kernel(p,state,jnp.asarray(idx),jnp.asarray(z),h))
                    for r in range(4):
                        g,raw=est.gradient(theta[r],None if method=="ula" else idx[r],return_raw_mean=True)
                        maxg=max(maxg,float(np.linalg.norm(np.asarray(parts[0])[r]-g)/(1+np.linalg.norm(g))))
                        np.testing.assert_allclose(np.asarray(parts[0])[r],g,atol=1e-9,rtol=1e-10)
                        np.testing.assert_allclose(np.asarray(parts[3])[r],raw,atol=1e-10,rtol=1e-10)
                        if geometry=="full":theta[r]=theta[r]-h*(np.asarray(p["P"])@g)+np.sqrt(2*h)*(np.asarray(p["L"])@z[r])
                        else:
                            D=metrics[r].update(raw) if geometry=="li2016" else 1.
                            theta[r]=theta[r]-h*D*g+np.sqrt(2*h*D)*z[r]
                        if geometry=="li2016":
                            maxv=max(maxv,float(np.max(np.abs(np.asarray(next_state[1])[r]-metrics[r].v))))
                            np.testing.assert_allclose(np.asarray(next_state[1])[r],metrics[r].v,atol=1e-10,rtol=1e-10)
                    maxstep=max(maxstep,float(np.max(np.abs(np.asarray(public)-theta))))
                    np.testing.assert_allclose(public,theta,atol=1e-9,rtol=1e-10)
                    saved.append(np.asarray(public));state=next_state
                values=np.asarray(ready(functional_chunk(jnp.asarray(saved),p["anchor"],p["transform"],jnp.asarray(f.slow_direction),jnp.asarray(f.prediction_rows))))
                np.testing.assert_allclose(values,np.sum(f(np.asarray(saved)),axis=0),atol=1e-8,rtol=1e-10)
                rows.append(dict(kind=kind,geometry=geometry,method=method,max_gradient_relative_error=maxg,max_state_absolute_error=maxstep,max_v_absolute_error=maxv))
        # Gaussian CV2=ULA for the same fixed batches/noise, independent of minibatch choice.
        p,_=prepare(X,y,anchor.theta,1.,kind,"cv2","full",f.transform,device=device)
        if kind=="gaussian":
            s1=s2=initial_state(p,4)
            for k in range(20):
                idx=jnp.asarray(np.stack([rng.choice(37,4,replace=False) for _ in range(4)]));z=jnp.asarray(rng.normal(size=(4,5)))
                s1,x,_=ready(fixed_input_kernel("cv2","full",kind)(p,s1,idx,z,.05))
                s2,yout,_=ready(fixed_input_kernel("ula","full",kind)(p,s2,idx,z,.05))
                np.testing.assert_allclose(x,yout,rtol=1e-10,atol=1e-9)
    # A failed lane remains dead and does not contaminate other lanes.
    p,_=prepare(X,y,anchor.theta,1.,kind,"cv2","full",f.transform,device=device)
    state=initial_state(p,4);idx=jnp.tile(jnp.arange(4),(4,1));z=jnp.zeros((4,5)).at[0,0].set(1e9)
    state,x,_=ready(fixed_input_kernel("cv2","full",kind)(p,state,idx,z,.01))
    assert not bool(state[2][0]) and bool(jnp.all(state[2][1:])) and int(state[3][0])==1
    state,x,_=ready(fixed_input_kernel("cv2","full",kind)(p,state,idx,jnp.zeros((4,5)),.01))
    assert np.isnan(np.asarray(x)[0]).all() and np.isfinite(np.asarray(x)[1:]).all() and int(state[3][0])==1
    # Random streams remain paired across methods; chunking preserves global iteration keys.
    p,_=prepare(X,y,anchor.theta,1.,kind,"cv2","full",f.transform,device=device)
    s=initial_state(p,4)
    s8,x8=ready(chunk_kernel("cv2","full",kind,4,8)(p,s,144100,.001))
    s4,x4=ready(chunk_kernel("cv2","full",kind,4,4)(p,s,144100,.001))
    s44,x44=ready(chunk_kernel("cv2","full",kind,4,4)(p,s4,144100,.001))
    np.testing.assert_allclose(np.concatenate([x4,x44]),x8,atol=1e-10,rtol=1e-10)
    report=dict(status="passed",device=str(device),device_kind=device.device_kind,platform=device.platform,jax=jax.__version__,
        x64=jax.config.x64_enabled,common_input_checks=rows,gaussian_cv2_ula_parity=True,failure_mask_and_no_revival=True,
        chunk_invariance=True,scope="JAX CPU validation is not GPU validation; require platform=gpu/A100 for remote gate")
    (output/"REPORT.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="common_input_checks"}),flush=True)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",required=True);parser.add_argument("--require-gpu",action="store_true")
    args=parser.parse_args();run(args.output,args.require_gpu)

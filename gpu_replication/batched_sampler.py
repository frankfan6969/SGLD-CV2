"""Independent JAX backend. Does not alter the CPU sampler or protocols.

All math float64; independent replica keys; distinct batch/noise purposes.
Functional evaluation is separate from timed sampling chunks, like CPU P1.
"""
from functools import lru_cache
from time import perf_counter
import numpy as np
import jax
jax.config.update("jax_enable_x64",True)
import jax.numpy as jnp
from jax import lax


def ready(tree):
    return jax.tree_util.tree_map(lambda x:x.block_until_ready(),tree)


def prepare(X,y,anchor,prior,kind,method,geometry,transform,*,noise_variance=1.,device=None):
    """Actual transfer/cache/metric setup, separately synchronized and timed."""
    start=perf_counter()
    X,y,anchor,transform=[jax.device_put(np.asarray(x,dtype=np.float64),device) for x in [X,y,anchor,transform]]
    ready((X,y,anchor,transform));transfer=perf_counter()-start;t=perf_counter()
    d=X.shape[1]
    if kind=="logistic":
        p=jax.nn.sigmoid(X@anchor);residual=p-y
        weights=p*(1-p) if method=="cv2" or geometry=="full" else jnp.zeros_like(y)
    elif kind=="gaussian":
        residual=(X@anchor-y)/noise_variance;weights=jnp.ones_like(y)/noise_variance
    else:raise ValueError("unknown likelihood")
    gradient_hat=X.T@residual if method in ["cv1","cv2"] else jnp.zeros(d,dtype=jnp.float64)
    H=(X.T*weights)@X if method=="cv2" or geometry=="full" else jnp.zeros((d,d),dtype=jnp.float64)
    P=jnp.eye(d,dtype=jnp.float64);L=P
    if geometry=="full":
        regularized=(H+H.T)/2+(prior+1e-10*prior)*jnp.eye(d)
        check=jnp.linalg.cholesky(regularized)
        P=jnp.linalg.solve(regularized,jnp.eye(d));P=(P+P.T)/2;L=jnp.linalg.cholesky(P)
        ready(check)
        if not np.isfinite(np.asarray(check)).all():raise ValueError("metric not SPD")
    params=dict(X=X,y=y,anchor=anchor,prior=jnp.asarray(prior,dtype=jnp.float64),noise_variance=jnp.asarray(noise_variance,dtype=jnp.float64),
        residual_hat=residual,weights_hat=weights,gradient_hat=gradient_hat,H_data=H,P=P,L=L,transform=transform)
    ready(params)
    return params,dict(transfer_seconds=transfer,cache_metric_seconds=perf_counter()-t,total_seconds=perf_counter()-start,
        scope="synchronized transfer+actual caches/metric; anchor optimization added by caller; no compile or sampling included")


def initial_state(params,replicas):
    d=params["X"].shape[1]
    return (jnp.broadcast_to(params["anchor"],(replicas,d)),jnp.zeros((replicas,d),dtype=jnp.float64),
            jnp.ones(replicas,dtype=bool),jnp.zeros(replicas,dtype=jnp.int32),jnp.asarray(0,dtype=jnp.int32),jnp.asarray(0,dtype=jnp.int64))


def gradient_parts(params,theta,indices,method,kind):
    X,y=params["X"],params["y"];R=theta.shape[0];d=theta.shape[1]
    prior=params["prior"]*theta;zeros=jnp.zeros((R,d),dtype=jnp.float64)
    if method=="ula":
        logits=theta@X.T
        residual=jax.nn.sigmoid(logits)-y if kind=="logistic" else (logits-y)/params["noise_variance"]
        data=residual@X
        return data+prior,zeros,zeros,data/X.shape[0]
    xb=X[indices];logits=jnp.einsum("rbd,rd->rb",xb,theta)
    residual=jax.nn.sigmoid(logits)-y[indices] if kind=="logistic" else (logits-y[indices])/params["noise_variance"]
    raw=jnp.einsum("rbd,rb->rd",xb,residual)/indices.shape[1]
    if method=="sgld":return X.shape[0]*raw+prior,zeros,zeros,raw
    remainder=residual-params["residual_hat"][indices]
    correction=zeros
    if method=="cv2":
        delta=theta-params["anchor"]
        remainder-=params["weights_hat"][indices]*jnp.einsum("rbd,rd->rb",xb,delta)
        correction=delta@params["H_data"].T
    random_remainder=(X.shape[0]/indices.shape[1])*jnp.einsum("rbd,rb->rd",xb,remainder)
    return params["gradient_hat"]+correction+random_remainder+prior,correction,random_remainder,raw


def advance(params,state,indices,normal,h,method,geometry,kind,alpha,epsilon):
    theta,v,live,failed_at,t,logical=state
    g,correction,remainder,raw=gradient_parts(params,theta,indices,method,kind)
    next_v=alpha*v+(1-alpha)*raw**2 if geometry=="li2016" else v
    if geometry=="full":
        drift=-h*(g@params["P"].T);noise=jnp.sqrt(2*h)*(normal@params["L"].T)
    else:
        D=1/(epsilon+jnp.sqrt(next_v)) if geometry=="li2016" else jnp.ones_like(theta)
        drift=-h*D*g;noise=jnp.sqrt(2*h*D)*normal
    proposal=theta+drift+noise
    radius=jnp.linalg.norm((proposal-params["anchor"])@params["transform"],axis=1)
    good=jnp.all(jnp.isfinite(proposal),axis=1)&jnp.isfinite(radius)&(radius<=100*jnp.sqrt(theta.shape[1]))
    new_live=live&good;first=live&~good
    failed_at=jnp.where(first,t+1,failed_at)
    # Dead lanes compute padded work but never update, replace or re-enter.
    updated=jnp.where(new_live[:,None],proposal,jnp.zeros_like(proposal))
    next_v=jnp.where(live[:,None],next_v,v)
    state=(updated,next_v,new_live,failed_at,t+1,logical+jnp.sum(live))
    public=jnp.where(new_live[:,None],updated,jnp.nan)
    return state,public,(g,correction,remainder,raw,drift,noise,radius)


@lru_cache(maxsize=64)
def fixed_input_kernel(method,geometry,kind,alpha=.99,epsilon=1e-5):
    def step(params,state,indices,normal,h):
        return advance(params,state,indices,normal,h,method,geometry,kind,alpha,epsilon)
    return jax.jit(step)


@lru_cache(maxsize=64)
def chunk_kernel(method,geometry,kind,batch_size,chunk=128,alpha=.99,epsilon=1e-5):
    def run(params,state,seed,h):
        R=state[0].shape[0];d=state[0].shape[1];N=params["X"].shape[0]
        root=jax.random.PRNGKey(seed)
        replica_keys=jax.vmap(lambda r:jax.random.fold_in(root,r))(jnp.arange(R,dtype=jnp.uint32))
        batch_keys=jax.vmap(lambda k:jax.random.fold_in(k,1))(replica_keys)
        noise_keys=jax.vmap(lambda k:jax.random.fold_in(k,2))(replica_keys)
        def body(carry,_):
            t=carry[4]
            nk=jax.vmap(lambda k:jax.random.fold_in(k,t))(noise_keys)
            normal=jax.vmap(lambda k:jax.random.normal(k,(d,),dtype=jnp.float64))(nk)
            if method=="ula":indices=jnp.zeros((R,batch_size),dtype=jnp.int32)
            else:
                bk=jax.vmap(lambda k:jax.random.fold_in(k,t))(batch_keys)
                indices=jax.vmap(lambda k:jax.random.choice(k,N,(batch_size,),replace=False))(bk)
            new,public,_=advance(params,carry,indices,normal,h,method,geometry,kind,alpha,epsilon)
            return new,public
        return lax.scan(body,state,None,length=chunk)
    return jax.jit(run)


@jax.jit
def functional_chunk(states,anchor,transform,slow_direction,prediction_rows):
    q=(states-anchor)@transform;slow=(states-anchor)@slow_direction
    values=jnp.concatenate([q,q*q,slow[...,None],(slow*slow)[...,None],jax.nn.sigmoid(states@prediction_rows.T)],axis=-1)
    return jnp.sum(values,axis=0)

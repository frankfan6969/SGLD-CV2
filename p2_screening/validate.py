"""Required independent checks before consuming P2 pilot seeds."""
import json,sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from kernels import subset,hmc_kernel,sample_kernel,logit_energy_gradient
from batched_sampler import prepare,initial_state,ready,advance

def validate(device):
    keys=jax.random.split(jax.random.PRNGKey(180001),24000)
    a=np.asarray(ready(jax.jit(jax.vmap(lambda k:subset(k,5,2)))(keys)))
    assert ((a>=0)&(a<5)).all() and (a[:,0]!=a[:,1]).all()
    pairs,c=np.unique(np.sort(a,axis=1),axis=0,return_counts=True)
    assert len(pairs)==10 and np.max(np.abs(c-2400))<220
    rng=np.random.default_rng(180002);X=rng.normal(size=(37,5));y=rng.binomial(1,.5,37)
    for method,geometry in [('cv1','full'),('cv2','full'),('ula','full'),('sgld','identity'),('cv2','li2016'),('cv1','li2016'),('sgld','li2016')]:
        p,_=prepare(X,y,np.zeros(5),1.,'logistic',method,geometry,np.eye(5),device=device)
        B=np.eye(5)*.3; hp=dict(X=p['X'],y=p['y'],anchor=p['anchor'],B=jnp.asarray(B)); zz=jnp.asarray(rng.normal(size=(3,5)))
        analytic=logit_energy_gradient(hp,zz)[1]; automatic=jax.grad(lambda q:logit_energy_gradient(hp,q)[0].sum())(zz)
        np.testing.assert_allclose(analytic,automatic,rtol=1e-10,atol=1e-10)
        state=initial_state(p,3);end,xx=ready(sample_kernel(method,geometry)(p,state,180003,.0001))
        for t in range(128):
            roots=jax.vmap(lambda r:jax.random.fold_in(jax.random.PRNGKey(180003),r))(jnp.arange(3,dtype=jnp.uint32))
            keys=jax.vmap(lambda k:jax.random.fold_in(k,t))(roots)
            idx=jnp.zeros((3,32),dtype=jnp.int32) if method=='ula' else jax.vmap(lambda k:subset(jax.random.fold_in(k,1),37,32))(keys)
            noise=jax.vmap(lambda k:jax.random.normal(jax.random.fold_in(k,2),(5,),dtype=jnp.float64))(keys)
            state,public,_=advance(p,state,idx,noise,.0001,method,geometry,'logistic',.99,1e-5)
            np.testing.assert_allclose(np.asarray(public),np.asarray(xx[t]),rtol=1e-9,atol=1e-9)
    state=(jnp.zeros((4,3)),jnp.full(4,.3),jnp.asarray(0));draws=[]
    for i in range(9):
        state,(z,acc,div)=ready(hmc_kernel(False,True)({},state,180004))
        if i:draws.append(np.asarray(z))
        assert not np.asarray(div).any()
    z=np.concatenate(draws).reshape(-1,3)
    assert np.max(np.abs(z.mean(0)))<.12 and np.max(np.abs(z.var(0)-1))<.15
    return dict(status='passed',device=str(device),x64=jax.config.x64_enabled,checks=['uniform unique Floyd subsets','fixed-stream scan vs independent step loop, seven arms','analytic vs autodiff logistic HMC gradient','Metropolis HMC Gaussian moments and no divergences'])
if __name__=='__main__':print(json.dumps(validate(jax.devices()[0]),indent=2))

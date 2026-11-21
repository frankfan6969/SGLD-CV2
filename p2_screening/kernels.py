"""P2 pilot kernels. The P1 estimator and transition algebra is reused unchanged."""
from functools import lru_cache
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from jax import lax
from batched_sampler import advance

def subset(key,N,b):
    # Floyd's algorithm: exact uniform subset, O(b²) work, independent of N.
    # Ordering is immaterial to all finite-sum estimators used here.
    def body(i,a):
        j=N-b+i;t=jax.random.randint(jax.random.fold_in(key,i),(),0,j+1,dtype=jnp.int32)
        v=jnp.where(jnp.any((jnp.arange(b)<i)&(a==t)),j.astype(jnp.int32),t)
        return a.at[i].set(v)
    return lax.fori_loop(0,b,body,jnp.full((b,),-1,dtype=jnp.int32))

@lru_cache(None)
def sample_kernel(method,geometry,b=32,alpha=.99,epsilon=1e-5):
    def run(p,state,seed,h):
        R,d=state[0].shape;N=p['X'].shape[0]
        roots=jax.vmap(lambda r:jax.random.fold_in(jax.random.PRNGKey(seed),r))(jnp.arange(R,dtype=jnp.uint32))
        def body(state,_):
            t=state[4]
            keys=jax.vmap(lambda k:jax.random.fold_in(k,t))(roots)
            idx=jnp.zeros((R,b),dtype=jnp.int32) if method=='ula' else jax.vmap(lambda k:subset(jax.random.fold_in(k,1),N,b))(keys)
            noise=jax.vmap(lambda k:jax.random.normal(jax.random.fold_in(k,2),(d,),dtype=jnp.float64))(keys)
            new,public,_=advance(p,state,idx,noise,h,method,geometry,'logistic',alpha,epsilon)
            return new,public
        return lax.scan(body,state,None,length=128)
    return jax.jit(run)

def logit_energy_gradient(p,z):
    theta=p['anchor']+z@p['B'].T;v=theta@p['X'].T
    U=jnp.sum(jnp.logaddexp(0.,v)-v*p['y'],axis=1)+.5*jnp.sum(theta**2,axis=1)
    g=((jax.nn.sigmoid(v)-p['y'])@p['X']+theta)@p['B']
    return U,g

@lru_cache(None)
def hmc_kernel(warmup,gaussian=False):
    def run(p,state,seed):
        z,eps,t=state
        def evaluate(q):return (.5*jnp.sum(q*q,axis=1),q) if gaussian else logit_energy_gradient(p,q)
        def body(state,_):
            z,eps,t=state;k=jax.random.fold_in(jax.random.PRNGKey(seed),t)
            momentum=jax.random.normal(jax.random.fold_in(k,1),z.shape,dtype=jnp.float64)
            e=eps*jax.random.uniform(jax.random.fold_in(k,2),(z.shape[0],),minval=.9,maxval=1.1)
            U,g=evaluate(z);q=z;v=momentum-.5*e[:,None]*g
            def leap(i,carry):
                q,v=carry;q=q+e[:,None]*v;_,g=evaluate(q)
                return q,v-e[:,None]*g*jnp.where(i==7,.5,1.)
            q,v=lax.fori_loop(0,8,leap,(q,v));Un,_=evaluate(q)
            error=Un+.5*jnp.sum(v*v,axis=1)-U-.5*jnp.sum(momentum*momentum,axis=1)
            prob=jnp.where(jnp.isfinite(error),jnp.exp(jnp.minimum(-error,0)),0.)
            accept=jax.random.uniform(jax.random.fold_in(k,3),prob.shape)<prob
            new=jnp.where(accept[:,None],q,z)
            if warmup:eps=jnp.clip(jnp.exp(jnp.log(eps)+.05/jnp.sqrt(1+t/10)*(prob-.8)),.001,1.)
            return (new,eps,t+1),(new,accept,~jnp.isfinite(error)|(jnp.abs(error)>100))
        return lax.scan(body,(z,eps,t),None,length=128)
    return jax.jit(run)

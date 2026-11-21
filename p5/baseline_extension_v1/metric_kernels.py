"""Fixed identity/diagonal/full metric ablation, full CV2 correction preserved."""
from functools import lru_cache
import jax
import jax.numpy as jnp
from jax import lax
from batched_sampler import gradient_parts
from kernels import subset
from extension_kernels import gradient_kernel
@lru_cache(None)
def metric_step(method,geometry,block=4096):
 grad=gradient_kernel(block)
 def step(p,s,indices,noise,h):
  theta,v,live,failed,t,logical=s;d=theta.shape[1];g=grad(p,theta) if method=='ula' else gradient_parts(p,theta,indices,method,'logistic')[0]
  if geometry=='full':proposal=theta-h*(g@p['P'].T)+jnp.sqrt(2*h)*(noise@p['L'].T)
  else:proposal=theta-h*g*p['D']+jnp.sqrt(2*h*p['D'])*noise
  radius=jnp.linalg.norm((proposal-p['anchor'])@p['transform'],axis=1);good=jnp.all(jnp.isfinite(proposal),axis=1)&jnp.isfinite(radius)&(radius<=100*jnp.sqrt(d));alive=live&good;failed=jnp.where(live&~good,t+1,failed);updated=jnp.where(alive[:,None],proposal,0.)
  return (updated,v,alive,failed,t+1,logical+jnp.sum(live)),jnp.where(alive[:,None],updated,jnp.nan)
 return jax.jit(step)
@lru_cache(None)
def metric_kernel(method,geometry,b,block=4096):
 step=metric_step(method,geometry,block)
 def run(p,state,seed,h):
  R,d=state[0].shape;N=p['X'].shape[0];roots=jax.vmap(lambda r:jax.random.fold_in(jax.random.PRNGKey(seed),r))(jnp.arange(R,dtype=jnp.uint32))
  def body(s,_):
   keys=jax.vmap(lambda k:jax.random.fold_in(k,s[4]))(roots);noise=jax.vmap(lambda k:jax.random.normal(jax.random.fold_in(k,2),(d,),dtype=jnp.float64))(keys);ix=jnp.zeros((R,1),dtype=jnp.int32) if method=='ula' else jax.vmap(lambda k:subset(jax.random.fold_in(k,1),N,b))(keys)
   return step(p,s,ix,noise,h)
  return lax.scan(body,state,None,length=128)
 return jax.jit(run)

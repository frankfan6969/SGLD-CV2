"""Exact data-blocked ULA, sharing original RNG/update/failure conventions."""
from functools import lru_cache
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from jax import lax
@lru_cache(None)
def gradient_kernel(block):
 def gradient(p,theta):
  X,y=p['X'],p['y'];N,d=X.shape
  if block>=N:return (jax.nn.sigmoid(theta@X.T)-y)@X+p['prior']*theta
  count=N//block
  def body(i,g):
   xx=lax.dynamic_slice(X,(i*block,0),(block,d));yy=lax.dynamic_slice(y,(i*block,),(block,))
   return g+(jax.nn.sigmoid(theta@xx.T)-yy)@xx
  g=lax.fori_loop(0,count,body,jnp.zeros_like(theta))
  if N%block:
   xx=X[count*block:];yy=y[count*block:];g=g+(jax.nn.sigmoid(theta@xx.T)-yy)@xx
  return g+p['prior']*theta
 return jax.jit(gradient)
@lru_cache(None)
def ula_step(block):
 grad=gradient_kernel(block)
 def step(p,state,normal,h):
  theta,v,live,failed,t,logical=state;proposal=theta-h*(grad(p,theta)@p['P'].T)+jnp.sqrt(2*h)*(normal@p['L'].T)
  radius=jnp.linalg.norm((proposal-p['anchor'])@p['transform'],axis=1);good=jnp.all(jnp.isfinite(proposal),axis=1)&jnp.isfinite(radius)&(radius<=100*jnp.sqrt(theta.shape[1]));alive=live&good;failed=jnp.where(live&~good,t+1,failed);updated=jnp.where(alive[:,None],proposal,0.)
  return (updated,v,alive,failed,t+1,logical+jnp.sum(live)),jnp.where(alive[:,None],updated,jnp.nan)
 return jax.jit(step)
@lru_cache(None)
def ula_kernel(block):
 step=ula_step(block)
 def run(p,state,seed,h):
  R,d=state[0].shape;roots=jax.vmap(lambda r:jax.random.fold_in(jax.random.PRNGKey(seed),r))(jnp.arange(R,dtype=jnp.uint32))
  def body(s,_):
   keys=jax.vmap(lambda k:jax.random.fold_in(k,s[4]))(roots);noise=jax.vmap(lambda k:jax.random.normal(jax.random.fold_in(k,2),(d,),dtype=jnp.float64))(keys)
   return step(p,s,noise,h)
  return lax.scan(body,state,None,length=128)
 return jax.jit(run)
@jax.jit
def sum_functionals(states,anchor,W):
 q=(states-anchor)@W;return jnp.sum(jnp.concatenate([q,q*q],axis=-1),axis=0)

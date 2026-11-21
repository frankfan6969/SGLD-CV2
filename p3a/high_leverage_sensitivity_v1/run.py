from pathlib import Path
import sys,json,time,hashlib
from functools import lru_cache
import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits
R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(R/'src'),str(R/'gpu_replication'),str(R/'p2_screening')]
import jax
import jax.numpy as jnp
from jax import lax
from kernels import subset
from batched_sampler import prepare,ready,advance
H=Path(__file__).parent;O=R/'exports/p3a_high_leverage_sensitivity_v1';D=R/'results/p3a_bank_jax_cpu_amendment1_validation_v1';A=R/'results/p3a_bank_t4_heldout_v1/frozen_bank_assets';c=json.loads((H/'PROTOCOL.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert all(sha(R/p)==v for p,v in c['input_hashes'].items());assert jax.default_backend()=='cpu' and jax.config.x64_enabled
save=lambda name,x:(O/name).write_text(json.dumps(x,indent=2)+'\n');start=time.perf_counter();suspect=c['suspect_observation'];a=np.load(A/'target.npz');X,y,anchor,W=[a[k] for k in ['X','y','anchor','W']];N,d=X.shape
@lru_cache(None)
def trigger_batch(n):
 @jax.jit
 def f(seed,lane,t):
  root=jax.random.fold_in(jax.random.PRNGKey(seed),lane);key=jax.random.fold_in(root,t-1);return subset(jax.random.fold_in(key,1),N,n)
 return f
@lru_cache(None)
def replay_kernel(n):
 def f(p,state,seed,stop,lane,h):
  roots=jax.vmap(lambda r:jax.random.fold_in(jax.random.PRNGKey(seed),r))(jnp.arange(512,dtype=jnp.uint32))
  initial=(state,jnp.zeros(d),jnp.zeros(n,dtype=jnp.int32),jnp.zeros(d),jnp.zeros(d),jnp.zeros(d),jnp.zeros(d),jnp.zeros(d),jnp.zeros(d),jnp.array(0.))
  def body(carry):
   s=carry[0];keys=jax.vmap(lambda k:jax.random.fold_in(k,s[4]))(roots);ix=jax.vmap(lambda k:subset(jax.random.fold_in(k,1),N,n))(keys);z=jax.vmap(lambda k:jax.random.normal(jax.random.fold_in(k,2),(d,),dtype=jnp.float64))(keys);new,public,parts=advance(p,s,ix,z,h,'cv2','full','logistic',.99,1e-5);g,cor,rem,raw,dr,noise,radius=parts
   return (new,s[0][lane],ix[lane],z[lane],g[lane],cor[lane],rem[lane],dr[lane],noise[lane],radius[lane])
  return lax.while_loop(lambda carry:carry[0][4]<stop,body,initial)
 return jax.jit(f)
with threadpool_limits(limits=1):
 p,_=prepare(X,y,anchor,1.,'logistic','cv2','full',W);P,L,Hd,ga=[np.asarray(p[k]) for k in ['P','L','H_data','gradient_hat']];p0=expit(X@anchor);w=p0*(1-p0);lev=np.sum((X@P)*X,axis=1);threshold=100*np.sqrt(d)
 events=[]
 for e in c['all_lane_failure_events']:
  ix=np.asarray(ready(trigger_batch(e['batch'])(e['seed'],e['lane'],e['failure_iteration'])));events.append(dict(**e,suspect_present=bool(suspect in ix),minibatch_indices=ix.tolist(),unconditional_inclusion_probability=e['batch']/N))
 save('all_failure_event_batches.json',events)
 records=[]
 for ci,case in enumerate(c['first_failure_cases']):
  if time.perf_counter()-start>c['budget_seconds']:raise TimeoutError('Frozen diagnostic budget')
  rid=f"n{case['batch']}_h{case['h']}_seed{case['seed']}";dest=O/f'case_{rid}.json'
  if dest.exists():records.append(json.loads(dest.read_text()));continue
  z=np.load(D/case['file']);failed=z['failed_at'];theta=np.nan_to_num(z['endpoints'],nan=0.);state=(jnp.asarray(theta),jnp.zeros((512,d)),jnp.asarray(failed==0),jnp.asarray(failed,dtype=jnp.int32),jnp.asarray(case['checkpoint'],dtype=jnp.int32),jnp.array(0,dtype=jnp.int64))
  out=ready(replay_kernel(case['batch'])(p,state,case['seed'],case['first_failure'],case['lane'],case['h']));final,t,ix,normal,g,cor,rem,dr,noise,radius=out;t,ix,normal,g,rem=[np.asarray(v) for v in [t,ix,normal,g,rem]];assert int(np.asarray(final[3])[case['lane']])==case['first_failure'];assert np.min(np.asarray(final[3])[np.asarray(final[3])>0])==case['first_failure'];expected=np.asarray(ready(trigger_batch(case['batch'])(case['seed'],case['lane'],case['first_failure'])));np.testing.assert_array_equal(ix,expected)
  delta=t-anchor;h=case['h'];n=len(ix);cor=Hd@delta;fixed=ga+cor+t;stochastic=(N/n)*X[ix]*(expit(X[ix]@t)-p0[ix]-w[ix]*(X[ix]@delta))[:,None];np.testing.assert_allclose(fixed+stochastic.sum(0),g,rtol=1e-9,atol=1e-7)
  def score(indices=None,omit=False):
   if indices is None:terms=stochastic.copy();indices=ix
   else:terms=(N/n)*X[indices]*(expit(X[indices]@t)-p0[indices]-w[indices]*(X[indices]@delta))[:,None]
   if omit:terms=terms.copy();terms[np.asarray(indices)==suspect]=0
   rr=terms.sum(0);gg=fixed+rr;drift=-h*(P@gg);noise=np.sqrt(2*h)*(L@normal);prop=t+drift+noise;r=float(np.linalg.norm((prop-anchor)@W));return dict(radius_after=r,passes_single_step_radius_gate=bool(np.isfinite(r) and r<=threshold),whitened_drift_norm=float(np.linalg.norm(drift@W)),preconditioned_stochastic_remainder_norm=float(np.linalg.norm(rr@L)))
  original=score();np.testing.assert_allclose(original['radius_after'],float(radius),rtol=1e-9,atol=1e-7);present=bool(suspect in ix);controls=[int(j) for j in c['ordinary_replacement_pool'] if j not in ix][:c['replacement_count_per_case']];interventions=[];shams=[];contribution=None
  if present:
   pos=int(np.flatnonzero(ix==suspect)[0]);v=stochastic[pos]@L;total=stochastic.sum(0)@L;contribution=dict(vector_norm=float(np.linalg.norm(v)),squared_norm_ratio_to_total=float(v@v/(total@total)),signed_projection_fraction=float(v@total/(total@total)),interpretation='Vector contributions need not be positive shares or sum as squared norms; cancellation is possible')
   for replacement in controls:
    altered=ix.copy();altered[pos]=replacement;assert len(set(altered))==n;interventions.append(dict(replacement=replacement,**score(altered)))
   if n>1:
    candidates=np.flatnonzero(ix!=suspect);pos2=int(candidates[np.argmin(lev[ix[candidates]])]);old=int(ix[pos2])
    for replacement in controls:
     altered=ix.copy();altered[pos2]=replacement;assert len(set(altered))==n;shams.append(dict(replaced_ordinary=old,replacement=replacement,**score(altered)))
  oldtrace=R/'exports/p3a_mechanism_diagnostic_v1'/f'replay_{rid}.npz';match=None
  if oldtrace.exists():
   zz=np.load(oldtrace);np.testing.assert_allclose(t,zz['theta_before'][-1],rtol=1e-10,atol=1e-10);np.testing.assert_array_equal(ix,zz['indices'][-1]);match=True
  record=dict(**case,reproduced_first_failure=True,original_trace_matches=match,original=original,suspect_present=present,suspect_vector_contribution=contribution,replace_suspect=interventions,sham_replace_ordinary=shams,omit_suspect_component=score(omit=True) if present else None,anchor_distance_before=float(np.linalg.norm(delta@W)),scope='Same pre-step state, h, P, Gaussian noise, anchor/correction; changed minibatch is diagnostic only, not an unbiased production sampler')
  np.savez_compressed(O/f'trigger_{rid}.npz',theta_before=t,indices=ix,normal=normal,stochastic_terms=stochastic,gradient=g)
  save(dest.name,record);records.append(record);save('first_failure_cases.json',records);print('REPLAY',ci+1,'/',len(c['first_failure_cases']),rid,'suspect',present,'radius',original['radius_after'],'replacement passes',sum(x['passes_single_step_radius_gate'] for x in interventions),'/',len(interventions),flush=True)
 refrows=[];refs=np.load(A/'scoring.npz')
 for name in ['first','second']:
  for i in c['reference_indices']:
   t=refs[name][i];delta=t-anchor;pp=expit(X@t);v={}
   for method,res in [('cv1',pp-p0),('cv2',pp-p0-w*(X@delta))]:
    rr=X*res[:,None];center=(rr-rr.mean(0))@L;sq=np.sum(center*center,1);total=float(sq.sum());v[method]=dict(noise_energy_total=total,suspect_centered_energy=float(sq[suspect]),suspect_fraction=float(sq[suspect]/total),largest_contributor=int(np.argmax(sq)),suspect_rank=int(np.sum(sq>sq[suspect])+1),energy_without_suspect_fixed_center=float(total-sq[suspect]))
   refrows.append(dict(reference=name,index=i,methods=v,ratio_cv2_cv1=v['cv2']['noise_energy_total']/v['cv1']['noise_energy_total'],ratio_without_suspect_fixed_center=v['cv2']['energy_without_suspect_fixed_center']/v['cv1']['energy_without_suspect_fixed_center'],scope='Centered sum component removed for attribution only, not a new target or resampled-data variance'))
 save('reference_noise_attribution.json',refrows)
 summary=dict(status='complete',runs_replayed=len(records),lane_failure_events=len(events),suspect_present_events=sum(x['suspect_present'] for x in events),suspect_present_first_failures=sum(x['suspect_present'] for x in records),replacement_cases=len([x for x in records if x['suspect_present']]),replacement_all_pass_cases=sum(bool(x['replace_suspect']) and all(z['passes_single_step_radius_gate'] for z in x['replace_suspect']) for x in records),sham_all_fail_cases=sum(bool(x['sham_replace_ordinary']) and not any(z['passes_single_step_radius_gate'] for z in x['sham_replace_ordinary']) for x in records),elapsed_seconds=time.perf_counter()-start,protocol_sha256=sha(H/'PROTOCOL.json'),threshold=threshold)
 save('summary.json',summary);print(json.dumps(summary),flush=True)

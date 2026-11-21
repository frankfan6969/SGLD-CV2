"""Independent JAX CPU float64 validation; fixed pilot candidates, no retuning."""
from pathlib import Path
import sys,os,json,time,platform,hashlib,fcntl,zipfile,traceback
from time import perf_counter
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'gpu_replication'),str(ROOT/'p2_screening')]
import run as base
import numpy as np
import jax
import jax.numpy as jnp
from threadpoolctl import threadpool_limits
from kernels import sample_kernel
HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/p3a_bank_jax_cpu_amendment1_validation_v1'
ASSETS=ROOT/'results/p3a_bank_t4_heldout_v1/frozen_bank_assets'
def arrays(path,**values):
 path=Path(path);tmp=path.with_name(path.name+'.tmp')
 with tmp.open('wb') as f:np.savez_compressed(f,**values)
 os.replace(tmp,path)

def main():
 c=json.loads((HERE/'PROTOCOL.json').read_text());resume='--resume' in sys.argv
 assert jax.default_backend()=='cpu' and jax.config.x64_enabled
 assert base.digest(__file__)==c['runner_sha256'] and base.digest(HERE/'PREFLIGHT.json')==c['preflight_sha256']
 assert json.loads((HERE/'PREFLIGHT.json').read_text())['status']=='passed'
 assert all(base.digest(ROOT/p)==h for p,h in c['input_hashes'].items())
 assert json.loads((ASSETS/'reference_pair.json').read_text())['reliable']
 if OUT.exists() and not resume:raise RuntimeError('Existing results; explicit --resume required, never overwrite')
 OUT.mkdir(parents=True,exist_ok=True);guard=(OUT/'.run.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if resume:assert json.loads((OUT/'PROTOCOL.json').read_text())==c
 else:
  for name in ['PROTOCOL.json','run.py','PREFLIGHT.json','VALIDATION.md']:(OUT/name).write_bytes((HERE/name).read_bytes())
 previous=json.loads((OUT/'manifest.json').read_text()) if (OUT/'manifest.json').exists() else {};prior_elapsed=previous.get('elapsed_seconds',0.)
 if previous.get('status')=='complete':raise RuntimeError('Already complete; no rerun')
 if prior_elapsed>=c['stage_seconds']:raise RuntimeError('Cumulative frozen stage cap exhausted')
 consumed=[]
 for folder in ['p3a_bank_t4_pilot_v1','p3a_bank_t4_selection_v1','p3a_bank_t4_heldout_v1','p3a_bank_jax_cpu_amendment1_pilot_v1']:
  consumed.extend(r['seed'] for r in json.loads((ROOT/'results'/folder/'summary.json').read_text()))
 assert set(consumed).isdisjoint(c['validation_seeds'])
 base.save(OUT/'seed_audit.json',dict(passed=True,old_seeds=sorted(set(consumed)),pilot_seeds=c['pilot_seeds'],validation_seeds=c['validation_seeds'],validation_consumed=True))
 base.save(OUT/'environment.json',dict(platform=platform.platform(),python=sys.version,jax=jax.__version__,devices=[str(d) for d in jax.devices()],x64=jax.config.x64_enabled,xla_flags=os.environ.get('XLA_FLAGS'),pid=os.getpid(),backend='JAX CPU float64',resume=resume))
 jobs=[dict(method=q['method'],batch=q['batch'],h=q['h'],seed=s,max_iterations=q['max_iterations']) for s in c['validation_seeds'] for q in c['candidates']]
 jobs += [dict(method='ula',batch=32,h=h,seed=s,max_iterations=c['ula_max_iterations']) for s in c['validation_seeds'] for h in c['ula_steps']]
 np.random.default_rng(c['schedule_seed']).shuffle(jobs);assert len(jobs)==c['runs'];base.save(OUT/'schedule.json',jobs)
 records=json.loads((OUT/'summary.json').read_text()) if (OUT/'summary.json').exists() else []
 a=np.load(ASSETS/'target.npz');X,y,anchor,H,W=[a[k] for k in ['X','y','anchor','H','W']];a=np.load(ASSETS/'scoring.npz');refs=[a['first'],a['second']];proj=a['projections'];meta=json.loads((ASSETS/'target_geometry.json').read_text());start=perf_counter();compiled={}
 def elapsed():return prior_elapsed+perf_counter()-start
 def manifest(status,**kwargs):base.save(OUT/'manifest.json',dict(status=status,runs_completed=len(records),total_runs=c['runs'],elapsed_seconds=elapsed(),validation_consumed=True,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kwargs))
 manifest('preparing')
 try:
  with threadpool_limits(limits=1):
   for job in jobs:
    m,n,h,seed=job['method'],job['batch'],job['h'],job['seed'];rid=f'{m}_n{n}_h{h}_seed{seed}';result=OUT/f'result_{rid}.json';progress=OUT/f'progress_{rid}.npz';progress_meta=OUT/f'progress_{rid}.json'
    if result.exists():
     r=json.loads(result.read_text())
     if not any((r['method'],r['batch'],r['h'],r['seed'])==(x['method'],x['batch'],x['h'],x['seed']) for x in records):records.append(r)
     continue
    if elapsed()>c['stage_seconds']:raise TimeoutError('Frozen cumulative stage cap')
    if (m,n) not in compiled:
     p,_=base.prepare(X,y,anchor,1.,'logistic',m,'full',W);s=base.ready(base.initial_state(p,c['replicas']));fn=sample_kernel(m,'full',n,.99);t=perf_counter();exe=fn.lower(p,s,c['initialization_seed'],h).compile();compile_s=perf_counter()-t;t=perf_counter();base.ready(exe(p,s,c['initialization_seed'],h));warmup_s=perf_counter()-t;compiled[m,n]=(exe,compile_s)
     base.save(OUT/f'initialization_{m}_n{n}.json',dict(compile_seconds=compile_s,unscored128_seconds=warmup_s,scope='Warmup excluded; actual shape-specific compile attributed to cold per run; not fresh-process cold timing'))
    t=perf_counter();p,setup=base.prepare(X,y,anchor,1.,'logistic',m,'full',W);state=base.ready(base.initial_state(p,c['replicas']));setup_s=perf_counter()-t;L=np.asarray(p['L']);LP=float(np.linalg.eigvalsh(L.T@H@L)[-1]);assert abs(LP-1)<1e-7
    exe,compile_s=compiled[m,n];sums=jnp.zeros((c['replicas'],2*len(H)));sampling=functional=evaluation=0.;k=0;points=[];timing_interrupted=False
    if resume and progress.exists():
     old=json.loads(progress_meta.read_text());saved=np.load(progress);state=tuple(jnp.asarray(saved[f'state{i}']) for i in range(6));sums=jnp.asarray(saved['functional_sums']);k=int(state[4]);points=old['checkpoints'];sampling=old['sampling_seconds'];functional=old['functional_seconds'];evaluation=old['evaluation_seconds'];setup_s+=old['setup_seconds'];compile_s+=old['compile_seconds'];timing_interrupted=True
    manifest('sampling',current=job,current_iteration=k)
    while k<job['max_iterations']:
     if elapsed()>c['stage_seconds'] or sampling>c['per_run_sampler_seconds']:break
     t=perf_counter();state,xx=base.ready(exe(p,state,seed,h));sampling+=perf_counter()-t;k=int(state[4]);failed=np.asarray(state[3]);t=perf_counter();sums=base.ready(sums+base.sum_functionals(xx,p['anchor'],p['transform']));functional+=perf_counter()-t
     if k in c['checkpoints']:
      t=perf_counter();q=base.score(np.asarray(xx[-1]),np.asarray(sums),k,refs,anchor,W,proj);evaluation+=perf_counter()-t;count=int((failed>0).sum())
      if count:q=dict(status='failed_ensemble',references=None)
      q.update(iteration=k,hK=h*k,failed_replicas=count,failure_iteration=int(failed[failed>0].min()) if count else None,warm_total_seconds=meta['anchor_seconds']+setup_s+sampling,cold_total_seconds=meta['anchor_seconds']+setup_s+sampling+compile_s);points.append(q)
      arrays(OUT/f'checkpoint_{rid}_K{k}.npz',endpoints=np.asarray(xx[-1]),functional_sums=np.asarray(sums),failed_at=failed)
     if k==128 or k%512==0 or k==job['max_iterations']:
      arrays(progress,**{f'state{i}':np.asarray(s) for i,s in enumerate(state)},functional_sums=np.asarray(sums));base.save(progress_meta,dict(checkpoints=points,sampling_seconds=sampling,functional_seconds=functional,evaluation_seconds=evaluation,setup_seconds=setup_s,compile_seconds=compile_s));manifest('sampling',current=job,current_iteration=k)
    failed=np.asarray(state[3]);r=dict(**job,LP=LP,replicas=c['replicas'],completed_iterations=k,checkpoints=points,failed_replicas=int((failed>0).sum()),failed_at=failed.tolist(),budget_censored=k<job['max_iterations'],setup=setup,setup_seconds=setup_s,compile_seconds=compile_s,sampling_seconds=sampling,functional_seconds=functional,evaluation_seconds=evaluation,timing_interrupted=timing_interrupted)
    base.save(result,r);records.append(r);base.save(OUT/'summary.json',records);manifest('sampling',current=job,current_iteration=k)
    print('JAX_CPU_AMENDMENT_VALIDATION',len(records),'/',c['runs'],rid,'K',k,'failed',r['failed_replicas'],flush=True)
   manifest('complete',scope='Independent CPU validation; pilot-fixed choices retained; no post-validation retuning or V100 timing claim')
 except Exception:
  manifest('failed_or_stage_censored',error=traceback.format_exc());raise
 finally:
  archive=OUT/'P3A_JAX_CPU_Validation_Amendment1.zip'
  with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
   for f in OUT.iterdir():
    if f.is_file() and f!=archive and not f.name.endswith('.tmp') and f.name!='.run.lock':z.write(f,f.name)
 print('JAX_CPU_VALIDATION_COMPLETE',str(archive),flush=True)
if __name__=='__main__':main()

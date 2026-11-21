"""Bounded post-heldout CPU tuning amendment. Independent results, resumable RNG."""
from pathlib import Path
import sys,json,os,hashlib,platform,fcntl,zipfile,time,traceback
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits,threadpool_info
from backend import ROOT,ASSETS,save,save_arrays,load_assets,prepare,transition,inputs,score
HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/p3a_bank_cpu_amendment1_pilot_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def shortlist(records,c):
 points=[];chosen=[]
 for m in ['cv1','cv2']:
  for n in c['batches']:
   candidate=[]
   for h in c['steps']:
    group=[r for r in records if (r['method'],r['batch'],r['h'])==(m,n,h)]
    valid=len(group)==2 and all(r['failed_replicas']==0 and not r['budget_censored'] and r['completed_iterations']==8192 for r in group)
    if not valid:continue
    for k in c['checkpoints']:
     ps=[next(p for p in r['checkpoints'] if p['iteration']==k) for r in group];v=np.array([[[z[t] for t in ['sw2','variance_nrmse']] for z in p['references']] for p in ps]).mean(0).max(0)
     q=dict(method=m,batch=n,h=h,iteration=k,sw2=float(v[0]),variance_nrmse=float(v[1]),warm_seconds=float(np.mean([p['warm_total_seconds'] for p in ps])),joint_score=float(max(v[0]/.3,v[1]/.5)));candidate.append(q)
   pareto=[q for q in candidate if not any(all(r[k]<=q[k] for k in ['warm_seconds','sw2','variance_nrmse']) and any(r[k]<q[k] for k in ['warm_seconds','sw2','variance_nrmse']) for r in candidate)]
   for q in candidate:q['pareto']=q in pareto
   points.extend(candidate)
   if not candidate:chosen.append(dict(method=m,batch=n,status='no_stable_configuration_within_frozen_search'));continue
   picked=[]
   def add(q):
    if q['h'] not in picked and len(picked)<3:picked.append(q['h'])
   joint=[q for q in pareto if q['sw2']<=.3 and q['variance_nrmse']<=.5]
   if joint:add(min(joint,key=lambda q:(q['warm_seconds'],q['joint_score'],q['h'],q['iteration'])))
   for metric in ['joint_score','variance_nrmse','sw2']:add(min(pareto,key=lambda q:(q[metric],q['warm_seconds'],q['h'],q['iteration'])))
   support={h:sum(q['h']==h for q in pareto) for h in set(q['h'] for q in pareto)}
   for h in sorted(support,key=lambda h:(-support[h],h)):
    if len(picked)<3 and h not in picked:picked.append(h)
   chosen.extend(dict(method=m,batch=n,h=h,status='pilot_candidate_only',precision='float64',cost_backend='CPU; must rebenchmark on V100') for h in picked)
 return points,chosen

def main():
 c=json.loads((HERE/'PROTOCOL.json').read_text());resume='--resume' in sys.argv
 assert sha(HERE/'backend.py')==c['backend_sha256'] and sha(__file__)==c['runner_sha256']
 assert json.loads((HERE/'PREFLIGHT.json').read_text())['status']=='passed'
 assert all(sha(ROOT/p)==h for p,h in c['input_hashes'].items())
 assert json.loads((ASSETS/'reference_pair.json').read_text())['reliable']
 if OUT.exists() and not resume:raise RuntimeError('Existing output: use explicit --resume; original results never overwritten')
 OUT.mkdir(parents=True,exist_ok=True);guard=(OUT/'.run.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if resume:assert json.loads((OUT/'PROTOCOL.json').read_text())==c
 else:
  for name in ['PROTOCOL.json','backend.py','run.py','PREFLIGHT.json']: (OUT/name).write_bytes((HERE/name).read_bytes())
 previous=json.loads((OUT/'manifest.json').read_text()) if (OUT/'manifest.json').exists() else {};prior_elapsed=previous.get('elapsed_seconds',0.)
 if previous.get('status')=='complete':raise RuntimeError('Already complete; no rerun')
 if prior_elapsed>=c['stage_seconds']:raise RuntimeError('Frozen cumulative stage budget exhausted; no silent extension')
 jobs=[dict(method=m,batch=n,h=h,seed=s,max_iterations=8192) for s in c['pilot_seeds'] for n in c['batches'] for m in ['cv1','cv2'] for h in c['steps']]
 jobs += [dict(method='ula',batch=32,h=h,seed=s,max_iterations=2048) for s in c['pilot_seeds'] for h in c['ula_steps']]
 np.random.default_rng(c['schedule_seed']).shuffle(jobs);assert len(jobs)==150
 save(OUT/'schedule.json',jobs);records=json.loads((OUT/'summary.json').read_text()) if (OUT/'summary.json').exists() else []
 consumed=[]
 for folder in ['p3a_bank_t4_pilot_v1','p3a_bank_t4_selection_v1','p3a_bank_t4_heldout_v1']:
  old=json.loads((ROOT/'results'/folder/'summary.json').read_text());consumed.extend(r['seed'] for r in old)
 assert set(consumed).isdisjoint(c['pilot_seeds']+c['validation_seeds_reserved'])
 save(OUT/'seed_audit.json',dict(passed=True,old_seeds=sorted(set(consumed)),pilot_seeds=c['pilot_seeds'],validation_consumed=False))
 save(OUT/'environment.json',dict(platform=platform.platform(),python=sys.version,numpy=np.__version__,threads=threadpool_info(),pid=os.getpid(),backend='CPU NumPy float64',resume=resume))
 start=perf_counter();a,refs,proj,meta=load_assets()
 def elapsed():return prior_elapsed+perf_counter()-start
 def manifest(status,**kwargs):save(OUT/'manifest.json',dict(status=status,runs_completed=len(records),total_runs=150,elapsed_seconds=elapsed(),validation_consumed=False,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kwargs))
 manifest('preparing')
 try:
  with threadpool_limits(limits=1):
   for job in jobs:
    rid=f"{job['method']}_n{job['batch']}_h{job['h']}_seed{job['seed']}";result=OUT/f'result_{rid}.json';progress=OUT/f'progress_{rid}.npz';progress_meta=OUT/f'progress_{rid}.json'
    if result.exists():
     r=json.loads(result.read_text())
     if not any((x['method'],x['batch'],x['h'],x['seed'])==(r['method'],r['batch'],r['h'],r['seed']) for x in records):records.append(r)
     continue
    if elapsed()>c['stage_seconds']:raise TimeoutError('Frozen cumulative stage budget exhausted')
    p,setup=prepare(a);R=c['replicas'];d=len(p['anchor']);theta=np.broadcast_to(p['anchor'],(R,d)).copy();live=np.ones(R,bool);failed=np.zeros(R,np.int32);sums=np.zeros((R,2*d));k=0;points=[];sampling=rng_s=functional=evaluation=0.;setup_s=setup['total_seconds'];timing_interrupted=False
    if progress.exists() and resume:
     old=json.loads(progress_meta.read_text());z=np.load(progress);theta,live,failed,sums=[z[x] for x in ['theta','live','failed','functional_sums']];k=int(z['iteration']);points=old['checkpoints'];sampling=old['sampling_seconds'];rng_s=old['rng_seconds'];functional=old['functional_seconds'];evaluation=old['evaluation_seconds'];setup_s+=old['setup_seconds'];timing_interrupted=True
    manifest('sampling',current=job,current_iteration=k)
    while k<job['max_iterations']:
     if elapsed()>c['stage_seconds'] or sampling+rng_s>c['per_run_sampler_seconds']:break
     block=k//128;t=perf_counter();ix,noise=inputs(job['seed'],block,R,job['batch'],len(p['X']),d,job['method']);rng_s+=perf_counter()-t
     for j in range(128):
      t=perf_counter();theta,live,failed,public=transition(p,theta,live,failed,None if ix is None else ix[j],noise[j],job['h'],job['method'],k);sampling+=perf_counter()-t;k+=1
      t=perf_counter();q=(public-p['anchor'])@p['W'];sums[:,:d]+=q;sums[:,d:]+=q*q;functional+=perf_counter()-t
     if k in c['checkpoints']:
      t=perf_counter();q=score(theta,sums,k,p,refs,proj,failed);evaluation+=perf_counter()-t;q.update(iteration=k,hK=job['h']*k,warm_total_seconds=meta['anchor_seconds']+setup_s+sampling+rng_s,legacy_anchor_attribution_seconds=meta['anchor_seconds']);points.append(q)
      save_arrays(OUT/f'checkpoint_{rid}_K{k}.npz',endpoints=public,functional_sums=sums,failed_at=failed)
     if k==128 or k%512==0 or k==job['max_iterations']:
      save_arrays(progress,theta=theta,live=live,failed=failed,functional_sums=sums,iteration=np.asarray(k));save(progress_meta,dict(checkpoints=points,sampling_seconds=sampling,rng_seconds=rng_s,functional_seconds=functional,evaluation_seconds=evaluation,setup_seconds=setup_s));manifest('sampling',current=job,current_iteration=k)
    r=dict(**job,replicas=R,LP=p['LP'],completed_iterations=k,checkpoints=points,failed_replicas=int((failed>0).sum()),failed_at=failed.tolist(),budget_censored=k<job['max_iterations'],setup=setup,setup_seconds=setup_s,sampling_seconds=sampling,rng_seconds=rng_s,functional_seconds=functional,evaluation_seconds=evaluation,timing_interrupted=timing_interrupted)
    save(result,r);records.append(r);save(OUT/'summary.json',records);manifest('sampling',current=job,current_iteration=k)
    print('CPU_AMENDMENT_PILOT',len(records),'/150',rid,'K',k,'failed',r['failed_replicas'],flush=True)
   points,candidates=shortlist(records,c);save(OUT/'configuration_points.json',points);save(OUT/'VALIDATION_CANDIDATES.json',candidates);manifest('complete',scope='Exploratory CPU tuning only; new validation must be frozen before using reserved seeds; no V100 cost claim')
 except Exception:
  manifest('failed_or_stage_censored',error=traceback.format_exc());raise
 finally:
  archive=OUT/'P3A_CPU_Tuning_Amendment1.zip'
  with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
   for f in OUT.iterdir():
    if f.is_file() and f!=archive and not f.name.endswith('.tmp') and f.name!='.run.lock':z.write(f,f.name)
 print('CPU_TUNING_COMPLETE_LOCAL_ARCHIVE',str(archive),flush=True)
if __name__=='__main__':main()

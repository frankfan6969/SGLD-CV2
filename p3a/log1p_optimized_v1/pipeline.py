from pathlib import Path
import sys,json,time,os,fcntl,subprocess,traceback,hashlib,zipfile,platform
from time import perf_counter
from types import SimpleNamespace
import numpy as np
from threadpoolctl import threadpool_limits
R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(R/'src'),str(R/'gpu_replication'),str(R/'p2_screening')]
import run as base
import jax
import jax.numpy as jnp
from kernels import sample_kernel
from batched_sampler import gradient_parts,advance
from sgld_cv.models import LogisticRegression
from sgld_cv.estimators import prepare_estimator
from sgld_cv.costs import CostLedger
H=Path(__file__).parent;O=R/'results/p3a_log1p_optimized_v1';E=R/'exports/p3a_log1p_optimized_v1';A=R/'exports/p3a_feature_sensitivity_v1/log1p_previous';c=json.loads((H/'PROTOCOL.json').read_text());save=base.save
assert jax.default_backend()=='cpu' and jax.config.x64_enabled;assert all(base.digest(R/p)==h for p,h in c['input_hashes'].items());guard=(O/'.run.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);start=perf_counter();compiled={};records=[];stage='preflight';used=set()
for folder in [R/'results',R/'exports']:
 for path in folder.rglob('*summary.json'):
  try:
   data=json.loads(path.read_text())
   if isinstance(data,list):used.update(int(r['seed']) for r in data if isinstance(r,dict) and 'seed' in r)
  except (ValueError,TypeError):pass
assert not (set(c['pilot']['seeds']+c['validation']['seeds'])&used);save(O/'seed_audit.json',dict(passed=True,pilot_seeds=c['pilot']['seeds'],validation_seeds=c['validation']['seeds'],prior_seeds=sorted(used)));save(O/'environment.json',dict(jax=jax.__version__,python=sys.version,platform=platform.platform(),devices=[str(x) for x in jax.devices()],pid=os.getpid(),x64=jax.config.x64_enabled))
assets=O/'frozen_assets';assets.mkdir(exist_ok=True)
for filename in ['target.npz','scoring.npz','metadata.json','reference_pair.json','reference_0.json','reference_1.json']:(assets/filename).write_bytes((A/filename).read_bytes())
a=np.load(A/'target.npz');X,y,anchor,W,Hess=[a[k] for k in ['X','y','anchor','W','H']];s=np.load(A/'scoring.npz');refs=[s['first'],s['second']];proj=s['projections'];meta=json.loads((A/'metadata.json').read_text())
for index in [0,1]:assert json.loads((A/f'reference_{index}.json').read_text())['reliable']
assert json.loads((A/'reference_pair.json').read_text())['reliable']
def progress(status='running',**kw):save(O/'manifest.json',dict(status=status,stage=stage,elapsed_seconds=perf_counter()-start,runs_completed=len(records),pid=os.getpid(),updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw))
def checktime():
 if perf_counter()-start>c['budget_seconds']:raise TimeoutError('Frozen3-hour pipeline cap')
def arrays(path,**kw):
 tmp=path.with_name(path.name+'.tmp')
 with tmp.open('wb') as f:np.savez_compressed(f,**kw)
 os.replace(tmp,path)
def ranks(points):
 if len(points)==0:return np.empty(0,dtype=int)
 p=np.asarray(points);left=np.arange(len(p));result=np.full(len(p),-1);rank=0
 while len(left):
  sub=p[left];dominates=np.all(sub[:,None,:]<=sub[None,:,:],axis=-1)&np.any(sub[:,None,:]<sub[None,:,:],axis=-1);front=~dominates.any(0);assert front.any();result[left[front]]=rank;left=left[~front];rank+=1
 return result

def runjob(phase,m,n,h,seed,replicas,maxk,checkpoints):
 rid=f'{phase}_{m}_n{n}_h{h:.12g}_seed{seed}';dest=O/(rid+'.json')
 if dest.exists():
  r=json.loads(dest.read_text());records.append(r);return r
 checktime();key=(m,n,replicas)
 if key not in compiled:
  p,_=base.prepare(X,y,anchor,1.,'logistic',m,'full',W);ss=base.ready(base.initial_state(p,replicas));fn=sample_kernel(m,'full',n,.99);t=perf_counter();exe=fn.lower(p,ss,c['seeds']['warmup'],h).compile();compile_s=perf_counter()-t;t=perf_counter();base.ready(exe(p,ss,c['seeds']['warmup'],h));warmup_s=perf_counter()-t;compiled[key]=(exe,compile_s);save(O/f'compile_{m}_n{n}_R{replicas}.json',dict(compile_seconds=compile_s,warmup128_seconds=warmup_s,cache='Actual shape-specific measured compilation, not fresh-process cold'))
 exe,compile_s=compiled[key];t=perf_counter();p,setup=base.prepare(X,y,anchor,1.,'logistic',m,'full',W);state=base.ready(base.initial_state(p,replicas));setup_s=perf_counter()-t;sums=jnp.zeros((replicas,2*len(anchor)));sampling=functional=evaluation=0.;points=[];last_k=0;censored=False
 for k in range(128,maxk+1,128):
  checktime()
  if sampling>c['per_run_sampler_seconds']:censored=True;break
  t=perf_counter();state,xx=base.ready(exe(p,state,seed,h));sampling+=perf_counter()-t;last_k=k;t=perf_counter();sums=base.ready(sums+base.sum_functionals(xx,p['anchor'],p['transform']));functional+=perf_counter()-t;failed=np.asarray(state[3])
  if k in checkpoints:
   t=perf_counter();q=base.score(np.asarray(xx[-1]),np.asarray(sums),k,refs,anchor,W,proj) if not np.any(failed) else dict(status='failed_ensemble',references=None);evaluation+=perf_counter()-t;q.update(iteration=k,failed_replicas=int(np.sum(failed>0)),warm_total_seconds=meta['anchor_seconds']+setup_s+sampling,cold_total_seconds=meta['anchor_seconds']+setup_s+sampling+compile_s);points.append(q);arrays(O/f'{rid}_K{k}.npz',endpoints=np.asarray(xx[-1]),functional_sums=np.asarray(sums),failed_at=failed)
  if k%512==0:progress(current=dict(phase=phase,method=m,batch=n,h=h,seed=seed,K=k))
 r=dict(id=rid,phase=phase,method=m,batch=n,h=h,seed=seed,replicas=replicas,max_iterations=maxk,completed_iterations=last_k,budget_censored=censored,failed_replicas=int(np.sum(np.asarray(state[3])>0)),failed_at=np.asarray(state[3]).tolist(),checkpoints=points,setup=setup,setup_seconds=setup_s,compile_seconds=compile_s,sampling_seconds=sampling,functional_seconds=functional,evaluation_seconds=evaluation,anchor_seconds_attributed=meta['anchor_seconds']);save(dest,r);records.append(r);save(O/'summary.json',records);print('RUN',len(records),rid,'failed',r['failed_replicas'],'K',last_k,flush=True);return r

def pilotpoints(runs):
 groups={}
 for r in runs:groups.setdefault((r['method'],r['batch'],r['h']),[]).append(r)
 points=[];stable=[]
 for key,rr in sorted(groups.items()):
  if len(rr)!=len(c['pilot']['seeds']) or any(r['failed_replicas'] or r['budget_censored'] for r in rr):continue
  stable.append(key)
  for k in c['pilot']['checkpoints']:
   ps=[next(p for p in r['checkpoints'] if p['iteration']==k) for r in rr];means=np.mean([[[z[t] for t in ['sw2','variance_nrmse']] for z in p['references']] for p in ps],0).max(0);points.append(dict(method=key[0],batch=key[1],h=key[2],K=k,warm=float(np.mean([p['warm_total_seconds'] for p in ps])),cold=float(np.mean([p['cold_total_seconds'] for p in ps])),sw2=float(means[0]),variance=float(means[1])))
 return points,stable
try:
 with threadpool_limits(limits=1):
  progress();model=LogisticRegression(X,y,prior_precision=1.);anc=SimpleNamespace(theta=anchor,cost=CostLedger());rng=np.random.default_rng(205000);theta=refs[0][:3];checks=[]
  for n in [1,16,256]:
   ix=np.array([rng.choice(len(X),n,replace=False) for _ in theta]);normal=rng.normal(size=theta.shape);pp={}
   for m in c['methods']:
    p,_=base.prepare(X,y,anchor,1.,'logistic',m,'full',W);pp[m]=p;est=prepare_estimator(model,m,anc);want=np.array([est.gradient(t,i) for t,i in zip(theta,ix)]);got=np.asarray(base.ready(gradient_parts(p,jnp.asarray(theta),jnp.asarray(ix),m,'logistic')[0]));np.testing.assert_allclose(got,want,rtol=1e-9,atol=1e-8);ss=list(base.initial_state(p,3));ss[0]=jnp.asarray(theta);_,public,_=base.ready(advance(p,tuple(ss),jnp.asarray(ix),jnp.asarray(normal),.003125,m,'full','logistic',.99,1e-5));expected=theta-.003125*(want@np.asarray(p['P']).T)+np.sqrt(.00625)*(normal@np.asarray(p['L']).T);np.testing.assert_allclose(public,expected,rtol=1e-9,atol=1e-9);np.testing.assert_allclose(np.asarray(p['L'])@np.asarray(p['L']).T,np.asarray(p['P']),rtol=1e-9,atol=1e-10);checks.append(dict(method=m,batch=n,max_gradient_error=float(abs(got-want).max())))
   np.testing.assert_array_equal(pp['cv1']['P'],pp['cv2']['P'])
  save(O/'PREFLIGHT.json',dict(status='passed',checks=checks,reference_gates=True,seed_audit=True));stage='base_pilot';pc=c['pilot'];jobs=[(m,n,h,seed) for m in c['methods'] for n in c['batch_sizes'] for h in c['base_h'] for seed in pc['seeds']];rng=np.random.default_rng(c['seeds']['schedule']);rng.shuffle(jobs);pilot=[runjob('pilot',m,n,h,seed,pc['replicas'],pc['max_iterations'],pc['checkpoints']) for m,n,h,seed in jobs]
  stage='bounded_refinement';points,stable=pilotpoints(pilot);extra=[]
  for m in c['methods']:
   for n in c['batch_sizes']:
    baseh=c['base_h'];ok=[h for h in baseh if (m,n,h) in stable];added=[]
    if not ok:added=[baseh[0]/4,baseh[0]/16]
    else:
     boundary=[(lo,hi) for lo,hi in zip(baseh[:-1],baseh[1:]) if lo in ok and hi not in ok]
     if boundary:lo,hi=boundary[-1];added.append(float(np.sqrt(lo*hi)))
     elif baseh[-1] in ok:added.append(baseh[-1]*2)
     sub=[p for p in points if p['method']==m and p['batch']==n];rk=ranks([[p[x] for x in ['warm','cold','sw2','variance']] for p in sub]);counts={h:sum(p['h']==h and r==0 for p,r in zip(sub,rk)) for h in baseh};gaps=sorted(zip(baseh[:-1],baseh[1:]),key=lambda z:(-(counts[z[0]]+counts[z[1]]),z[0]))
     for lo,hi in gaps:
      mid=float(np.sqrt(lo*hi))
      if mid not in added:added.append(mid)
      if len(added)>=2:break
    extra.extend((m,n,h) for h in added[:2])
  save(O/'FROZEN_REFINEMENT.json',dict(configurations=[dict(method=m,batch=n,h=h) for m,n,h in extra],scope='Single frozen pilot-dependent round before new configurations; no additional rounds'))
  jobs=[(m,n,h,seed) for m,n,h in extra for seed in pc['seeds']];rng.shuffle(jobs)
  for m,n,h,seed in jobs:pilot.append(runjob('pilot',m,n,h,seed,pc['replicas'],pc['max_iterations'],pc['checkpoints']))
  save(O/'pilot_summary.json',pilot);points,stable=pilotpoints(pilot);stage='candidate_freeze';selected=[]
  for m in c['methods']:
   sub=[p for p in points if p['method']==m];globalrank=ranks([[p[x] for x in ['warm','cold','sw2','variance']] for p in sub]);rankmap={(p['batch'],p['h'],p['K']):int(r) for p,r in zip(sub,globalrank)}
   for n in c['batch_sizes']:
    local=[p for p in sub if p['batch']==n];localrank=ranks([[p[x] for x in ['warm','cold','sw2','variance']] for p in local]);lmap={(p['h'],p['K']):int(r) for p,r in zip(local,localrank)};hs=sorted(set(p['h'] for p in local));scores=[]
    for h in hs:
     rs=[rankmap[n,h,k] for k in pc['checkpoints']];lr=[lmap[h,k] for k in pc['checkpoints']];scores.append((h,(-sum(r==0 for r in rs),min(rs),-sum(r==0 for r in lr),-sum(1/(1+r) for r in rs),h)))
    for h,score in sorted(scores,key=lambda z:z[1])[:2]:selected.append(dict(method=m,batch=n,h=h,pilot_selection_score=list(score)))
  if not selected:raise RuntimeError('No stable candidates; report failure without more tuning')
  extension=[]
  for q in selected:
   pp=[p for p in points if (p['method'],p['batch'],p['h'])==(q['method'],q['batch'],q['h'])];last=next(p for p in pp if p['K']==2048);prev=next(p for p in pp if p['K']==1024)
   if last['variance']>.20 and last['variance']<=.9*prev['variance']:extension.append(dict(**q,variance1024=prev['variance'],variance2048=last['variance']))
  horizon=4096 if extension else 2048;freeze=dict(candidates=selected,max_iterations=horizon,extension_trigger_cases=extension,missing_stable_method_batch=[dict(method=m,batch=n) for m in c['methods'] for n in c['batch_sizes'] if not any(q['method']==m and q['batch']==n for q in selected)],validation_seeds=c['validation']['seeds']);save(O/'FROZEN_CANDIDATES.json',freeze);print('FROZEN_CANDIDATES',len(selected),'horizon',horizon,flush=True)
  stage='independent_validation';vc=c['validation'];jobs=[(q,seed) for seed in vc['seeds'] for q in selected];rng.shuffle(jobs);validation=[]
  for q,seed in jobs:validation.append(runjob('validation',q['method'],q['batch'],q['h'],seed,vc['replicas'],horizon,[k for k in vc['checkpoints'] if k<=horizon]))
  save(O/'validation_summary.json',validation)
 stage='analysis';progress();env=dict(os.environ,MPLCONFIGDIR='/private/tmp/cv2_mpl');subprocess.run([sys.executable,str(H/'analyze.py')],check=True,env=env);subprocess.run(['/Library/Frameworks/Python.framework/Versions/3.10/bin/python3',str(H/'plot.py')],check=True,env=env);stage='archiving';progress()
 for p in H.iterdir():
  if p.is_file() and p.suffix in ['.py','.json']:(E/p.name).write_bytes(p.read_bytes())
 sourcefiles=[p for p in O.rglob('*') if p.is_file() and not p.name.startswith('.') and p.suffix not in ['.zip','.tmp']];exportfiles=[p for p in E.iterdir() if p.is_file() and p.suffix!='.zip' and p.name!='checksums.json'];checks={'raw/'+str(p.relative_to(O)):base.digest(p) for p in sourcefiles};checks.update({'review/'+p.name:base.digest(p) for p in exportfiles});save(E/'checksums.json',checks);archive=E/'P3A_Log1p_Optimized_v1.zip'
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  for p in sourcefiles:z.write(p,'raw/'+str(p.relative_to(O)))
  for p in exportfiles+[E/'checksums.json']:z.write(p,'review/'+p.name)
 with zipfile.ZipFile(archive) as z:assert z.testzip() is None
 stage='complete';progress(status='complete',pilot_runs=len(pilot),validation_runs=len(validation),archive=str(archive),archive_sha256=base.digest(archive));status=json.loads((R/'p3a/STATUS.json').read_text());status.update(stage='Log1p optimized comparison complete; report and archive verified',log1p_optimized_manifest=str((O/'manifest.json').relative_to(R)),log1p_optimized_report=str((E/'REPORT.md').relative_to(R)),log1p_optimized_archive=str(archive.relative_to(R)),log1p_optimized_archive_sha256=base.digest(archive),next_step='Review optimized Log1p Bank results; no further Bank tuning or P3B launched.');save(R/'p3a/STATUS.json',status);print('FULL_PIPELINE_COMPLETE',archive,flush=True)
except Exception:
 error=traceback.format_exc();progress(status='failed_or_budget_censored',error=error)
 (E/'REPORT.md').write_text('# Log1p optimized comparison: incomplete\n\nPipeline stopped at stage '+stage+'. Completed raw results are preserved. No unsupported efficiency conclusion is made.\n\n```\n'+error+'\n```\n')
 status=json.loads((R/'p3a/STATUS.json').read_text());status.update(stage='Log1p optimized pipeline stopped; incomplete report saved',log1p_optimized_report=str((E/'REPORT.md').relative_to(R)),log1p_optimized_manifest=str((O/'manifest.json').relative_to(R)));save(R/'p3a/STATUS.json',status)
 raise

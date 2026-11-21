"""Bounded P2 target screening on the existing A100; pilot evidence only."""
from pathlib import Path
import sys,os,json,hashlib,time,zipfile,urllib.request,io,csv,traceback,shutil,fcntl
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'gpu_replication'),str(ROOT/'p2_screening')]
import numpy as np
import jax
import jax.numpy as jnp
from scipy.special import expit
from threadpoolctl import threadpool_limits
from sgld_cv.models import LogisticRegression
from sgld_cv.anchors import find_anchor
from sgld_cv.diagnostics import chain_diagnostics
from sgld_cv.references import validate_reference_pair
from sgld_cv.transport import AccuracyMetric
from batched_sampler import prepare,initial_state,ready,gradient_parts
from kernels import hmc_kernel,sample_kernel
from validate import validate
OUT=ROOT/'results/p2_screening_v1'
DRIVE=Path('/content/drive/MyDrive/CV2_Research/P2_Screening_2026-10-15')

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)));os.replace(tmp,path)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sync_impl():
    errors=[];count=0
    for p in OUT.rglob('*'):
        if not p.is_file() or p.name.endswith('.tmp'):continue
        dst=DRIVE/p.relative_to(OUT);dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists() and dst.stat().st_size==p.stat().st_size and digest(dst)==digest(p):count+=1;continue
        try:
            tmp=dst.with_name(dst.name+'.uploading');shutil.copyfile(p,tmp)
            if digest(tmp)!=digest(p):raise IOError('backup checksum mismatch')
            os.replace(tmp,dst);count+=1
        except Exception as exc:errors.append(dict(file=str(p),error=str(exc)))
    save(DRIVE/'BACKUP_STATUS.json',dict(files=count,errors=errors,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    if errors:print('BACKUP_RETRY_REQUIRED',errors,flush=True)

def sync():
    with (ROOT/'.drive_sync.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        try:sync_impl()
        finally:fcntl.flock(guard,fcntl.LOCK_UN)

def data(spec):
    if spec['kind']=='bank':
        p=OUT/'bank_source.zip'
        with urllib.request.urlopen('https://archive.ics.uci.edu/static/public/222/bank%2Bmarketing.zip',timeout=120) as r:p.write_bytes(r.read())
        z=zipfile.ZipFile(p)
        if not any(n.endswith('bank-full.csv') for n in z.namelist()):
            member=next(n for n in z.namelist() if n.split('/')[-1]=='bank.zip');z=zipfile.ZipFile(io.BytesIO(z.read(member)))
        member=next(n for n in z.namelist() if n.endswith('bank-full.csv'))
        rows=list(csv.DictReader(io.StringIO(z.read(member).decode()),delimiter=';'));assert len(rows)==45211
        cols=[];names=[];trans={}
        for key in rows[0]:
            if key=='y':continue
            values=[r[key] for r in rows]
            try:
                v=np.array(values,dtype=float);mu=float(v.mean());sd=float(v.std()) or 1.
                cols.append((v-mu)/sd);names.append(key);trans[key]=dict(mean=mu,scale=sd)
            except ValueError:
                levels=sorted(set(values));trans[key]=dict(levels=levels,drop_reference=True)
                for lev in levels[1:]:cols.append(np.array(values)==lev);names.append(key+'='+lev)
        cols.append(np.ones(len(rows)));names.append('intercept')
        return np.column_stack(cols).astype(float),np.array([r['y']=='yes' for r in rows],dtype=float),dict(source_sha256=digest(p),features=names,preprocessing=trans,duration_included=True,scope='posterior benchmark, not a prospective prediction benchmark')
    rng=np.random.default_rng(181000);N,d=spec['N'],spec['d']
    Z=rng.normal(size=(N,d));Q,_=np.linalg.qr(rng.normal(size=(d,d)))
    scales=np.geomspace(1,spec['kappa']**-.5,d)
    X=(Z*scales)@Q.T if spec['geometry']=='correlated' else Z*scales
    beta=np.ones(d);beta/=np.std(X@beta);beta*=spec['signal']
    u=np.random.default_rng(181001).random(N);y=(u<expit(X@beta)).astype(float)
    return X,y,dict(true_theta=beta.tolist(),design_kappa=spec['kappa'],signal=spec['signal'],geometry=spec['geometry'],paired_design_seed=181000,label_uniform_seed=181001)

def reference(X,y,anchor,B,seed,deadline):
    p=dict(X=jnp.asarray(X),y=jnp.asarray(y),anchor=jnp.asarray(anchor),B=jnp.asarray(B));state=(jax.random.normal(jax.random.PRNGKey(seed),(4,len(anchor)),dtype=jnp.float64)*1.5,jnp.full(4,.25),jnp.asarray(0))
    draws=[];accepted=[];divergences=[];start=perf_counter()
    for block in range(10):
        if perf_counter()>deadline:raise TimeoutError('reference pilot budget exceeded')
        state,(z,acc,div)=ready(hmc_kernel(block<2)(p,state,seed))
        if block>=2:draws.append(np.asarray(z).transpose(1,0,2));accepted.append(np.asarray(acc));divergences.append(np.asarray(div))
    white=np.concatenate(draws,axis=1);theta=anchor+white@B.T
    diag=chain_diagnostics(theta)
    diag.update(acceptance=np.concatenate(accepted).mean(0).tolist(),divergences=np.concatenate(divergences).sum(0).tolist(),seconds=perf_counter()-start,kind='full-data MH HMC, fixed anchor whitening; warmup 256, retained 1024/chain; approximate pilot reference')
    diag['reliable']=bool(diag['reliable'] and min(diag['acceptance'])>.5 and sum(diag['divergences'])==0)
    return theta,diag

def score(endpoints,sums,k,refs,anchor,W,proj):
    if not np.isfinite(endpoints).all():return dict(status='failed_ensemble',sw2=None,variance_nrmse=None)
    metric=AccuracyMetric('standardized_sliced_w2',anchor,W,proj);d=len(anchor)
    avg=sums/k;var=avg[:,d:]-avg[:,:d]**2;values=[]
    for ref in refs:
        q=(ref-anchor)@W;v=q.var(0)
        rmse=float(np.mean(np.sqrt(np.mean(((var-v)/v)**2,axis=0))))
        values.append(dict(sw2=metric.distance(endpoints,ref),variance_nrmse=rmse))
    return dict(status='ok',references=values)

@jax.jit
def sum_functionals(states,anchor,W):
    q=(states-anchor)@W
    return jnp.sum(jnp.concatenate([q,q*q],axis=-1),axis=0)

def audit_gradient(X,y,anchor,W,states,seed):
    rng=np.random.default_rng(seed);R=len(states);idx=np.array([[rng.choice(len(X),32,replace=False) for r in range(R)] for b in range(32)])
    exact=(expit(states@X.T)-y)@X+states;output={};inv=np.linalg.inv(W)
    for method in ['sgld','cv1','cv2']:
        p,_=prepare(X,y,anchor,1.,'logistic',method,'full',W)
        f=jax.jit(jax.vmap(lambda ix:gradient_parts(p,jnp.asarray(states),ix,method,'logistic')[0]))
        g=np.asarray(ready(f(jnp.asarray(idx))));error=g-exact
        output[method]=dict(raw_mse_mean=float(np.mean(np.sum(error**2,axis=-1))),raw_mse_median=float(np.median(np.sum(error**2,axis=-1))),whitened_mse_mean=float(np.mean(np.sum((error@inv)**2,axis=-1))))
    return output

def target(spec,config,deadline):
    name=spec['id'];dest=OUT/name;dest.mkdir();t=perf_counter();X,y,meta=data(spec);model=LogisticRegression(X,y);anchor=find_anchor(model);H=model.full_hessian(anchor.theta)
    eig,V=np.linalg.eigh(H);W=(V*np.sqrt(eig))@V.T;B=np.linalg.inv(W)
    np.savez_compressed(dest/'target.npz',X=X,y=y,anchor=anchor.theta,H=H,W=W)
    meta.update(N=len(X),d=X.shape[1],posterior_anchor_kappa=float(eig[-1]/eig[0]),anchor_diagnostics=anchor.diagnostic if hasattr(anchor,'diagnostic') else str(anchor),anchor_seconds=anchor.cost.snapshot()['total_seconds'],target_sha256=digest(dest/'target.npz'),geometry_LP={g:float(np.linalg.eigvalsh((np.eye(len(H)) if g=='identity' else np.diag(1/np.sqrt(np.diag(H))))@H@(np.eye(len(H)) if g=='identity' else np.diag(1/np.sqrt(np.diag(H)))))[-1]) for g in ['identity','diagonal']},full_LP=1.)
    save(dest/'target_geometry.json',meta);sync();print('TARGET',name,'N',len(X),'d',X.shape[1],'kappa',meta['posterior_anchor_kappa'],flush=True)
    refs=[];diags=[]
    for j in range(2):
        samples,diag=reference(X,y,anchor.theta,B,182000+100*config['targets'].index(spec)+j,min(deadline,perf_counter()+600))
        refs.append(samples);diags.append(diag);np.savez_compressed(dest/f'reference_{j}.npz',samples=samples);save(dest/f'reference_{j}.json',diag);sync()
    pair=validate_reference_pair(*refs,*diags);save(dest/'reference_pair.json',pair)
    pooled=[r.reshape(-1,len(H)) for r in refs];states=pooled[0][np.linspace(0,len(pooled[0])-1,24,dtype=int)]
    save(dest/'gradient_mse.json',dict(reference_passed=pair['reliable'],diagnostic=audit_gradient(X,y,anchor.theta,W,states,183000)))
    if not pair['reliable']:
        save(dest/'SCREENING_STATUS.json',dict(status='reference_gate_failed',interpretation='No posterior accuracy ranking accepted. Reference requires separate refinement; pilot results preserved.'));sync();return
    rng=np.random.default_rng(184000);proj=rng.normal(size=(len(H),128));proj/=np.linalg.norm(proj,axis=0)
    scoring=[r[np.random.default_rng(184001+j).choice(len(r),256,replace=False)] for j,r in enumerate(pooled)]
    metric=AccuracyMetric('standardized_sliced_w2',anchor.theta,W,proj)
    floor=[metric.distance(pooled[0][np.random.default_rng(184100+j).choice(len(pooled[0]),64,replace=False)],scoring[1]) for j in range(16)]
    save(dest/'metric.json',dict(whitening='fixed symmetric sqrt(anchor posterior Hessian), pilot only',reference_floor_mean=float(np.mean(floor)),projections=128,SW2_absolute_certification=False));np.savez_compressed(dest/'scoring.npz',first=scoring[0],second=scoring[1],projections=proj)
    arms=[dict(method=m,geometry='full',coefficient=c,alpha=.99) for m in ['cv1','cv2','ula'] for c in [.05,.1,.2]]
    arms += [dict(method=m,geometry=g,coefficient=c,alpha=a) for m,g,c,a in [('sgld','identity',.05,.99),('cv1','identity',.05,.99),('cv2','identity',.05,.99),('cv1','diagonal',.1,.99),('cv2','diagonal',.1,.99),('sgld','li2016',.01,.9),('sgld','li2016',.04,.99),('cv1','li2016',.01,.99),('cv2','li2016',.01,.99)]]
    jobs=[(arm,s) for arm in arms for s in config['pilot_seeds']];np.random.default_rng(185000).shuffle(jobs)
    compile_cache={};records=[]
    for job,(arm,seed) in enumerate(jobs):
        if perf_counter()>deadline:save(dest/'SCREENING_STATUS.json',dict(status='budget_censored',completed_runs=job));break
        m,g,c,alpha=arm['method'],arm['geometry'],arm['coefficient'],arm['alpha'];setup_start=perf_counter();p,setup=prepare(X,y,anchor.theta,1.,'logistic',m,'full' if g=='diagonal' else g,W)
        if g=='diagonal':
            D=1/(np.diag(H)+1e-10);p['P']=jnp.diag(jnp.asarray(D));p['L']=jnp.diag(jnp.sqrt(jnp.asarray(D)));ready(p)
        LP=1. if g=='full' else meta['geometry_LP'].get(g,float(eig[-1]))
        if g=='li2016':
            ix=np.random.default_rng(185001).choice(len(X),32,replace=False);raw=(expit(X[ix]@anchor.theta)-y[ix])@X[ix]/32
            D=1/(1e-5+np.sqrt((1-alpha)*raw**2));LP=float(np.linalg.eigvalsh(np.sqrt(D)[:,None]*H*np.sqrt(D)[None,:])[-1])
        h=c/LP;state=ready(initial_state(p,64));setup_seconds=perf_counter()-setup_start
        key=(m,g,alpha);kernel=sample_kernel(m,'full' if g=='diagonal' else g,32,alpha)
        if key not in compile_cache:
            tt=perf_counter();exe=kernel.lower(p,state,seed,h).compile();compile_cache[key]=(exe,perf_counter()-tt)
        exe,compile_seconds=compile_cache[key];sums=jnp.zeros((64,2*len(H)));sampling=evaluation=0.;checkpoints=[];cap=False
        for block in range(16):
            if perf_counter()>deadline or sampling>120:cap=True;break
            tt=perf_counter();state,xx=ready(exe(p,state,seed,h));sampling+=perf_counter()-tt
            tt=perf_counter();sums=ready(sums+sum_functionals(xx,p['anchor'],p['transform']));evaluation+=perf_counter()-tt;k=int(state[4])
            if k in [128,512,2048]:
                endpoints=np.asarray(xx[-1]);avg=np.asarray(sums);value=score(endpoints,avg,k,scoring,anchor.theta,W,proj)
                value.update(iteration=k,warm_total_seconds=meta['anchor_seconds']+setup_seconds+sampling,cold_total_seconds=meta['anchor_seconds']+setup_seconds+sampling+compile_seconds)
                checkpoints.append(value)
        rid=f"{g}_{m}_c{c}_a{alpha}_seed{seed}";np.savez_compressed(dest/f'states_{rid}.npz',endpoints=np.asarray(xx[-1]),functional_sums=np.asarray(sums),failed_at=np.asarray(state[3]))
        record=dict(arm=arm,seed=seed,h=h,LP=LP,replicas=64,checkpoints=checkpoints,failed_replicas=int((~np.asarray(state[2])).sum()),completed_iterations=int(state[4]),setup_seconds=setup_seconds,sampler_seconds=sampling,compile_seconds=compile_seconds,functional_seconds=evaluation,budget_censored=cap)
        save(dest/f'result_{rid}.json',record);records.append(record);sync();print(name,job+1,'/',len(jobs),rid,'failed',record['failed_replicas'],flush=True)
    else:save(dest/'SCREENING_STATUS.json',dict(status='complete',completed_runs=len(records),scope='pilot target screening only; no heldout superiority or optimized frontier claim'))
    save(dest/'summary.json',records);sync()

def bootstrap_audit():
    rng=np.random.default_rng(186000);R=128;projection=rng.normal(size=(10,32));projection/=np.linalg.norm(projection,axis=0)
    ref=rng.normal(size=(256,10))@projection;ref=np.sort(ref,axis=0);q=np.repeat(np.arange(R),2)
    def distance(x):return float(np.sqrt(np.mean((np.sort(x,axis=0)[q]-ref)**2)))
    records=[]
    for trial in range(24):
        x=rng.normal(size=(R,10))@projection;point=distance(x);b=np.array([distance(x[rng.integers(R,size=R)]) for _ in range(199)])
        records.append(dict(point=point,bootstrap_mean=float(b.mean()),lower=float(np.quantile(b,.025)),upper=float(np.quantile(b,.975))))
    mean_finite_cloud=np.mean([distance(rng.normal(size=(R,10))@projection) for _ in range(512)])
    save(OUT/'SW2_bootstrap_audit.json',dict(scope='Independent Gaussian diagnostic; no replacement of P1 inference and no new P2 certification rule',population_SW2=0.,fixed_reference=True,records=records,finite_cloud_mean_mc=float(mean_finite_cloud),bootstrap_mean_shift=float(np.mean([r['bootstrap_mean']-r['point'] for r in records])),interval_fraction_containing_finite_cloud_mean=float(np.mean([r['lower']<=mean_finite_cloud<=r['upper'] for r in records]))))

def main():
    if not Path('/content/drive/MyDrive').is_dir():raise RuntimeError('Drive must remain mounted before starting P2')
    device=jax.devices('gpu')[0]
    if 'A100' not in device.device_kind:raise RuntimeError('Original A100 required')
    lock=json.loads((ROOT/'p2_screening/lock.json').read_text());assert all(digest(ROOT/p)==h for p,h in lock['files'].items())
    OUT.mkdir(parents=True,exist_ok=False);DRIVE.mkdir(parents=True,exist_ok=True)
    c=json.loads((ROOT/'p2_screening/PROTOCOL.json').read_text());save(OUT/'PROTOCOL.json',c);save(OUT/'source.lock.json',lock)
    save(OUT/'manifest.json',dict(status='validating',device=device.device_kind,jax=jax.__version__,x64=jax.config.x64_enabled,stage='P2 bounded screening',heldout_consumed=False));sync()
    try:
        with threadpool_limits(limits=1):
            save(OUT/'validation.json',validate(device));sync();start=perf_counter();bootstrap_audit();sync()
            for spec in c['targets']:
                if perf_counter()-start>7200:raise TimeoutError('P2 screening 2h cap')
                save(OUT/'manifest.json',dict(status='running',target=spec['id'],elapsed_seconds=perf_counter()-start,heldout_consumed=False))
                try:target(spec,c,min(start+7200,perf_counter()+1200))
                except Exception:save(OUT/(spec['id']+'_failure.json'),dict(error=traceback.format_exc(),classification='pilot failure; no silent survivor selection'));sync()
            save(OUT/'manifest.json',dict(status='complete',elapsed_seconds=perf_counter()-start,heldout_consumed=False,scope='Screening finished; reference gates and failures determine eligibility. Final P2 configuration still requires freeze.'))
    except Exception:
        save(OUT/'manifest.json',dict(status='failed',error=traceback.format_exc()));raise
    finally:
        with zipfile.ZipFile(OUT/'P2_Screening_Results.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in OUT.rglob('*'):
                if p.is_file() and p.name!='P2_Screening_Results.zip':z.write(p,str(p.relative_to(OUT)))
        sync()
    print('P2_SCREENING_COMPLETE_AND_BACKED_UP',DRIVE,flush=True)
if __name__=='__main__':main()

from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parents[2];D=R/'results/p3a_bank_jax_cpu_amendment1_validation_v1';A=R/'results/p3a_bank_t4_heldout_v1/frozen_bank_assets';O=Path(__file__).parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
runs=json.loads((D/'summary.json').read_text()); locations=[]
for ref in ['first','second']:
 for i in [0,32,64,96,128,160,192,224]: locations.append(dict(group='reference',reference=ref,index=i))
locations.insert(0,dict(group='anchor'))
for method in ['cv1','cv2']:
 for n,h in [(32,.003125),(256,.00625)]:
  for k in [128,8192]:
   for lane in [0,127,255,511]:locations.append(dict(group='trajectory',file=f'checkpoint_{method}_n{n}_h{h}_seed203800_K{k}.npz',lane=lane,method=method,batch=n,h=h,K=k))
for i in [0,64,128,192]:
 for scale in [2.,4.]:locations.append(dict(group='radial_stress',reference='first',index=i,scale=scale))
failed=sorted([x for x in runs if x['failed_replicas']],key=lambda x:(min(t for t in x['failed_at'] if t),x['batch'],x['h'],x['seed']))[:4]
for r in failed:
 t=min(v for v in r['failed_at'] if v);k=max(v for v in [128,512,2048] if v<t);lane=r['failed_at'].index(t)
 locations.append(dict(group='failure_previous_checkpoint',file=f"checkpoint_cv2_n{r['batch']}_h{r['h']}_seed{r['seed']}_K{k}.npz",lane=lane,batch=r['batch'],h=r['h'],seed=r['seed'],K=k,failure_iteration=t,scope='Last stored checkpoint before first failure, not immediate pre-failure state'))
files=['gpu_replication/batched_sampler.py','p2_screening/kernels.py','src/sgld_cv/estimators.py','src/sgld_cv/models.py','src/sgld_cv/preconditioners.py','src/sgld_cv/samplers.py','p3a/jax_cpu_validation_v1/run.py',str((A/'target.npz').relative_to(R)),str((A/'scoring.npz').relative_to(R)),str((D/'summary.json').relative_to(R))]
files+=list(dict.fromkeys(str((D/x['file']).relative_to(R)) for x in locations if 'file' in x))
c=dict(name='P3A bounded implementation and gradient-noise mechanism diagnostic v1',status='frozen before diagnostic computation; exploratory after P3A validation',backend='JAX CPU float64 plus independent NumPy/SciPy float64',batch_sizes=[1,4,16,32,64,256],locations=locations,seed=204100,monte_carlo=dict(locations=['reference:first:0','reference:second:0'],batch_sizes=[4,32],draws=1024,paired=True),exact_formula='N*(N-n)/n * mean_i ||(r_i-mean(r)) L||^2 where P=L L^T; uniform without replacement',metrics=['raw gradient noise MSE','preconditioned gradient noise MSE','preconditioned exact gradient norm','whitened anchor distance','per-location CV2/CV1 ratio'],reporting='Primary locations: independent references; trajectory and radial/failure locations descriptive. No algorithm ranking based only on its own trajectory. Anchor both zero. Fixed finite locations, not new posterior draws. No formal efficiency threshold changes.',audit=dict(methods=['cv1','cv2','ula'],step_sizes=[.003125,.00625],rtol=1e-9,atol=1e-8,noise='Verify L L^T=P and identical fixed-normal update; no empirical covariance test needed',cost='Reconstruct every checkpoint warm/cold from anchor+setup+sampling+compile; setup subcomponents aggregated in existing records, acknowledge historical anchor timing attribution'),scope='No new sampling grid, no P3B. Preconditioning ablation deferred pending diagnosis.',input_hashes={f:sha(R/f) for f in files})
(O/'PROTOCOL.json').write_text(json.dumps(c,indent=2)+'\n');print('Frozen',len(locations),'locations',sha(O/'PROTOCOL.json'))

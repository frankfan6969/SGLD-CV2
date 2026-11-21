"""Exact finite-population raw/preconditioned noise at comparable locations."""
from pathlib import Path
import json
import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits
R=Path(__file__).resolve().parents[2];H=Path(__file__).parent;c=json.loads((H/'PROTOCOL.json').read_text())
def run_mechanism(N):
 D=R/'results/p3b_n_scaling_v1'/f'N{N}';A=D/'target';a=np.load(A/'target.npz');X,y,anchor,W,B,L=[a[k] for k in ['X','y','anchor','W','B','L']];s=np.load(A/'scoring.npz');d=X.shape[1];rng=np.random.default_rng(c['mechanism']['direction_seed']);dirs=rng.normal(size=(16,d));dirs/=np.linalg.norm(dirs,axis=1)[:,None];locations=[dict(group='anchor',id='anchor',theta=anchor)]
 for pool,name in enumerate(['first','second']):
  for index in [0,16,32,48,64,80,96,112,128,144,160,176,192,208,224,240]:locations.append(dict(group='reference',id=f'pool{pool}_{index}',theta=s[name][index]))
 for scale in [.5,1.,2.]:
  for index,u in enumerate(dirs):locations.append(dict(group='controlled_shell',id=f'scale{scale}_dir{index}',scale=scale,direction=index,theta=anchor+B@u*(scale*np.sqrt(d))))
 p0=expit(X@anchor);weights=p0*(1-p0);rows=[]
 with threadpool_limits(limits=1):
  for loc in locations:
   theta=loc['theta'];delta=theta-anchor;prob=expit(X@theta);residuals=dict(sgld=prob-y,cv1=prob-p0,cv2=prob-p0-weights*(X@delta));base={k:v for k,v in loc.items() if k!='theta'}
   for method,res in residuals.items():
    rr=X*res[:,None];center=rr-rr.mean(0);white=rr@L;wc=center@L;rawenergy=np.sum(center*center,1);we=np.sum(wc*wc,1);rawnorm=np.linalg.norm(rr,axis=1);wn=np.linalg.norm(white,axis=1);order=np.argsort(we);total=float(we.sum());basis=dict(raw_noise_population_energy_mean=float(rawenergy.mean()),preconditioned_noise_population_energy_mean=float(we.mean()),max_raw_residual_norm=float(rawnorm.max()),q99_raw_residual_norm=float(np.quantile(rawnorm,.99)),q999_raw_residual_norm=float(np.quantile(rawnorm,.999)),max_preconditioned_residual_norm=float(wn.max()),q99_preconditioned_residual_norm=float(np.quantile(wn,.99)),q999_preconditioned_residual_norm=float(np.quantile(wn,.999)),largest_noise_observation=int(order[-1]),largest_noise_share=float(we[order[-1]]/total) if total else 0.,top10_noise_share=float(we[order[-10:]].sum()/total) if total else 0.)
    for n in [4,16,64]:
     factor=N*N*(N-n)/(n*(N-1));rows.append(dict(N=N,method=method,batch=n,raw_anchor_distance=float(np.linalg.norm(delta)),white_anchor_distance=float(np.linalg.norm(delta@W)),raw_mse=factor*basis['raw_noise_population_energy_mean'],preconditioned_mse=factor*basis['preconditioned_noise_population_energy_mean'],**base,**basis))
 (D/'mechanism_locations.json').write_text(json.dumps(rows,indent=2)+'\n');summ=[]
 for method in ['sgld','cv1','cv2']:
  for group,scale in [('reference',None),('controlled_shell',.5),('controlled_shell',1.),('controlled_shell',2.)]:
   q=[r for r in rows if r['method']==method and r['batch']==16 and r['group']==group and (scale is None or r['scale']==scale)];summ.append(dict(N=N,method=method,group=group,scale=scale,locations=len(q),raw_mse_mean=float(np.mean([r['raw_mse'] for r in q])),preconditioned_mse_mean=float(np.mean([r['preconditioned_mse'] for r in q])),max_raw_remainder_across_locations=float(max(r['max_raw_residual_norm'] for r in q)),median_location_max_raw_remainder=float(np.median([r['max_raw_residual_norm'] for r in q])),max_preconditioned_remainder_across_locations=float(max(r['max_preconditioned_residual_norm'] for r in q)),median_largest_noise_share=float(np.median([r['largest_noise_share'] for r in q])),mean_raw_anchor_distance=float(np.mean([r['raw_anchor_distance'] for r in q]))))
 (D/'mechanism_summary.json').write_text(json.dumps(summ,indent=2)+'\n');return summ
if __name__=='__main__':
 import sys
 run_mechanism(int(sys.argv[1]))

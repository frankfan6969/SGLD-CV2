"""Presentation-only full41-threshold heatmap; no new statistical estimators."""
from pathlib import Path
import os,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
R=Path(__file__).resolve().parents[2];O=Path(os.environ.get('P3B_REPORT_INPUT',str(R/'exports/p3b_n_scaling_v1')));a=json.loads((O/'analysis.json').read_text());Ns=a['protocol']['data']['N'];thresholds=sorted(a['protocol']['optimized']['threshold_sweep']['variance'],reverse=True);sweeps=a['optimized_thresholds'];arrays={}
for cost in ['warm','cold']:
 for typ,key in [('Point-estimate','point_speedup_cv1_over_cv2'),('Simultaneous-supported','supported_point_speedup')]:
  m=np.full((len(Ns),len(thresholds)),np.nan)
  for i,N in enumerate(Ns):
   for j,t in enumerate(thresholds):
    q=next((q for q in sweeps if q['N']==N and q['cost']==cost and q['variance_threshold']==t),None)
    if q and q[key] is not None and q[key]>0:m[i,j]=np.log2(q[key])
  arrays[cost,typ]=m
finite=np.concatenate([q[np.isfinite(q)] for q in arrays.values()]);vmax=max(1.,float(np.ceil(abs(finite).max()))) if len(finite) else 1.;fig,ax=plt.subplots(2,2,figsize=(13,6.5),layout='constrained');cmap=plt.get_cmap('RdBu').copy();cmap.set_bad('#d9d9d9')
for i,cost in enumerate(['warm','cold']):
 for j,typ in enumerate(['Point-estimate','Simultaneous-supported']):
  im=ax[i,j].imshow(arrays[cost,typ],cmap=cmap,norm=TwoSlopeNorm(vmin=-vmax,vcenter=0,vmax=vmax),aspect='auto',interpolation='nearest');ax[i,j].set(title=f'{cost.title()}: {typ}',yticks=np.arange(len(Ns)),yticklabels=[f'N={n:,}' for n in Ns],xticks=list(range(0,len(thresholds),5)),xticklabels=[f'{thresholds[k]:.2f}' for k in range(0,len(thresholds),5)],xlabel='Variance NRMSE target (stricter →)')
cbar=fig.colorbar(im,ax=ax.ravel().tolist(),shrink=.9);cbar.set_label('log2(CV1 time / CV2 time); positive favors CV2');fig.suptitle('P3B full validated-grid efficiency sweep; whitened SW2≤0.3');fig.supxlabel('Gray: missing or no finite ratio because one/both methods did not attain. No infinite speedups.',fontsize=9);fig.savefig(O/'N_threshold_heatmap.png',dpi=180);fig.savefig(O/'N_threshold_heatmap.pdf');print('FULL_THRESHOLD_HEATMAP_GENERATED')

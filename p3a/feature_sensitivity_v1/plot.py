from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[2];O=R/'exports/p3a_feature_sensitivity_v1';rows=json.loads((O/'mechanism_locations.json').read_text());fig,ax=plt.subplots(1,2,figsize=(11,4.5),layout='constrained');metrics=['max_preconditioned_residual_norm','exact_preconditioned_noise_mse'];radii=[1.,2.,4.]
for v,col in [('original','tab:red'),('log1p_previous','tab:blue'),('winsor999_previous','tab:green')]:
 for a,key in zip(ax,metrics):
  med=[];low=[];high=[]
  for rad in radii:
   vals=[r['methods']['cv2'][key] for r in rows if r['version']==v and r['radius_scale']==rad];med.append(np.median(vals));low.append(np.quantile(vals,.1));high.append(np.quantile(vals,.9))
  a.plot(radii,med,'o-',label=v,color=col);a.fill_between(radii,low,high,color=col,alpha=.12);a.set(yscale='log',xlabel='Whitened radius / sqrt(d)');a.grid(alpha=.2)
ax[0].set_ylabel('Max CV2 preconditioned residual norm');ax[1].set_ylabel('Exact CV2 gradient-noise MSE (n=32)');ax[0].legend(fontsize=8);fig.suptitle('Reference-free mechanism screen: median and 10–90% across fixed directions');fig.savefig(O/'mechanism_screen.png',dpi=180);fig.savefig(O/'mechanism_screen.pdf')

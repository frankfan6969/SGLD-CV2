from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[2];O=R/'exports/p3a_high_leverage_sensitivity_v1';cases=json.loads((O/'first_failure_cases.json').read_text());refs=json.loads((O/'reference_noise_attribution.json').read_text());s=json.loads((O/'summary.json').read_text());fig,ax=plt.subplots(1,2,figsize=(13,5),layout='constrained');x=np.arange(len(cases));ax[0].scatter(x,[c['original']['radius_after'] for c in cases],c='tab:red',marker='x',s=45,label='Original trigger update')
for i,c in enumerate(cases):
 for key,col,label in [('replace_suspect','tab:blue','Replace29182:16 controls'),('sham_replace_ordinary','tab:orange','Sham: keep29182')]:
  q=[z['radius_after'] for z in c[key]]
  if q:ax[0].vlines(i,min(q),max(q),color=col,linewidth=2);ax[0].scatter([i],[np.median(q)],c=col,s=16,label=label if i==next(j for j,z in enumerate(cases) if z[key]) else None)
ax[0].axhline(s['threshold'],c='k',ls='--',lw=1,label='Failure radius');ax[0].set(yscale='log',xlabel='First-failure case (20 runs)',ylabel='Whitened radius after identical-noise update');ax[0].legend(fontsize=8);ax[0].grid(alpha=.2)
for m,col in [('cv1','tab:blue'),('cv2','tab:orange')]:ax[1].plot(np.arange(16),[z['methods'][m]['suspect_fraction']*100 for z in refs],'o-',color=col,label=m.upper())
ax[1].set(xlabel='Fixed reference location index',ylabel='29182 share of centered noise energy (%)',ylim=(-2,102));ax[1].legend();ax[1].grid(alpha=.2);fig.suptitle('Bank high-leverage sensitivity: fixed-input single-step intervention',fontsize=13);fig.savefig(O/'sensitivity.png',dpi=180);fig.savefig(O/'sensitivity.pdf')

import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
out=Path('exports/p3a_pilot_review_v1');a=json.loads((out/'review.json').read_text());rows=a['configurations']
fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,key,title in zip(axes,['sw2','variance_nrmse'],['Endpoint standardized SW₂','Ergodic variance NRMSE']):
 for m,col in [('cv1','#1769aa'),('cv2','#e2711d'),('ula','#333333')]:
  q=[x for x in rows if x['method']==m and x['valid']]
  ax.scatter([x['warm_seconds'] for x in q],[x[key] for x in q],s=16,c=col,alpha=.5,label=m.upper())
  # Descriptive pilot frontier across all valid checkpoints, excluding every configuration with any failed seed across the full pilot.
  best=float('inf');front=[]
  for p in sorted(q,key=lambda x:x['warm_seconds']):
   if p[key]<best:front.append(p);best=p[key]
  ax.plot([x['warm_seconds'] for x in front],[x[key] for x in front],color=col)
 ax.set(xscale='log',yscale='log',xlabel='Setup-inclusive warm time (T4 seconds)',ylabel=title)
 ax.grid(alpha=.2);ax.legend()
axes[0].axhline(.3,color='gray',ls=':',lw=1);axes[1].axhline(.5,color='gray',ls=':',lw=1)
fig.suptitle('P3A tuning pilot — 2 seeds ×32 replicas; descriptive grid frontier')
fig.tight_layout();fig.savefig(out/'pilot_frontiers.png',dpi=180);fig.savefig(out/'pilot_frontiers.pdf');plt.close(fig)

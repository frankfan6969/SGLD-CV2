from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[2];O=R/'exports/p3a_mechanism_diagnostic_v1';rows=json.loads((O/'gradient_noise.json').read_text());rows=[x for x in rows if x['batch_size']==32 and x['ratio_cv2_cv1'] is not None];ref=[x for x in rows if x['group']=='reference'];fig,ax=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
for group,color,label in [('reference','tab:blue','Independent reference (16)'),('trajectory','tab:green','Stored trajectories (32)'),('radial_stress','tab:orange','Radial stress (8)'),('failure_previous_checkpoint','tab:red','Earlier failure checkpoints (4)')]:
 q=[x for x in rows if x['group']==group];ax[0].scatter([x['anchor_distance'] for x in q],[x['ratio_cv2_cv1'] for x in q],c=color,label=label,alpha=.8,s=35)
ax[0].set(yscale='log',xlabel='Whitened distance from anchor',ylabel='Exact conditional noise MSE: CV2 / CV1');ax[0].axhline(1,c='k',ls='--',lw=1);ax[0].legend(fontsize=8);ax[0].grid(alpha=.2)
x=np.arange(16);ax[1].plot(x,[r['cv1_preconditioned_mse'] for r in ref],'o-',label='CV1');ax[1].plot(x,[r['cv2_preconditioned_mse'] for r in ref],'o-',label='CV2');ax[1].set(yscale='log',xlabel='Fixed reference location index',ylabel='Preconditioned gradient-noise MSE (n=32)');ax[1].legend();ax[1].grid(alpha=.2);fig.suptitle('Bank: typical variance reduction with local reversals',fontsize=14)
fig.savefig(O/'noise_diagnostic.png',dpi=180);fig.savefig(O/'noise_diagnostic.pdf');plt.close(fig)

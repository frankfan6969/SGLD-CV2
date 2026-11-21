from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[2];O=R/'exports/p3b_n_scaling_v1';a=json.loads((O/'analysis.json').read_text());rows=[q for q in a['mechanism'] if q['group']=='reference'];fig,ax=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
for m,col in [('sgld','tab:gray'),('cv1','tab:blue'),('cv2','tab:orange')]:
 q=sorted([q for q in rows if q['method']==m],key=lambda q:q['N'])
 for x,key,label in zip(ax,['raw_mse_mean','preconditioned_mse_mean'],['Raw gradient-noise MSE','Preconditioned gradient-noise MSE']):
  if q:x.plot([r['N'] for r in q],[r[key] for r in q],'o-',color=col,label=m.upper())
  x.set(xscale='log',yscale='log',xlabel='Dataset size N',ylabel=label);x.grid(alpha=.2)
ax[0].legend();fig.suptitle('P3B: exact conditional noise, fixed n=16,32 reference locations per N');fig.savefig(O/'gradient_noise_scaling.png',dpi=180);fig.savefig(O/'gradient_noise_scaling.pdf');plt.close(fig)
fig,ax=plt.subplots(1,3,figsize=(13,4),layout='constrained');q=sorted(a['target_metadata'],key=lambda q:q['N']);ax[0].plot([r['N'] for r in q],[r['max_x_norm'] for r in q],'o-',label='max||x||');ax[0].set(xscale='log',xlabel='N',ylabel='Largest design row norm');
for group,scale,label,col in [('reference',None,'Reference locations','tab:blue'),('controlled_shell',1.,'Matched white radius sqrt(d)','tab:orange')]:
 q=sorted([q for q in a['mechanism'] if q['method']=='cv2' and q['group']==group and q['scale']==scale],key=lambda q:q['N'])
 for x,key in zip(ax[1:],['max_raw_remainder_across_locations','max_preconditioned_remainder_across_locations']):
  if q:x.plot([r['N'] for r in q],[r[key] for r in q],'o-',label=label,color=col)
  x.set(xscale='log',yscale='log',xlabel='N');x.legend(fontsize=8)
ax[1].set_ylabel('Max raw CV2 residual');ax[2].set_ylabel('Max preconditioned CV2 residual');[x.grid(alpha=.2) for x in ax];fig.suptitle('Observation extremes versus concentration: maxima over observations/locations');fig.savefig(O/'remainder_scaling.png',dpi=180);fig.savefig(O/'remainder_scaling.pdf');plt.close(fig)
fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for i,cost in enumerate(['warm','cold']):
 for threshold,col in [(.5,'tab:blue'),(.4,'tab:orange'),(.3,'tab:green'),(.2,'tab:red')]:
  qs=sorted([q for q in a['optimized_thresholds'] if q['cost']==cost and q['variance_threshold']==threshold],key=lambda q:q['N'])
  for m,ls in [('cv1','--'),('cv2','-')]:
   nn=[];tt=[]
   for q in qs:
    choice=q['methods'][m]['point_choice']
    if choice:nn.append(q['N']);tt.append(choice[cost+'_seconds'])
   if nn:ax[i,0].plot(nn,tt,'o'+ls,color=col,label=f'{m.upper()} ε={threshold}')
  yy=[q['point_speedup_cv1_over_cv2'] if q['point_speedup_cv1_over_cv2'] is not None else np.nan for q in qs];ax[i,1].plot([q['N'] for q in qs],yy,'o-',color=col,label=f'ε={threshold}')
 ax[i,0].set(xscale='log',yscale='log',xlabel='N',ylabel=f'Minimum grid {cost} seconds');ax[i,1].set(xscale='log',xlabel='N',ylabel=f'CV1/CV2 {cost} time ratio');ax[i,1].axhline(1,c='k',ls=':',lw=1)
 for x in ax[i]:x.legend(fontsize=7,ncol=2);x.grid(alpha=.2)
fig.suptitle('P3B optimized validated-grid time-to-accuracy; whitened SW2≤0.3');fig.savefig(O/'time_to_accuracy_scaling.png',dpi=180);fig.savefig(O/'time_to_accuracy_scaling.pdf');plt.close(fig)

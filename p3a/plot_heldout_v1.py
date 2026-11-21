from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path('exports/p3a_heldout_review_v1');a=json.loads((OUT/'analysis.json').read_text());rows=a['configuration_checkpoints'];colors={'cv1':'#1769aa','cv2':'#e2711d','ula':'#333333'}
for cost in ['warm','cold']:
 fig,axes=plt.subplots(1,2,figsize=(11.5,4.5))
 for ax,key,label in zip(axes,['sw2','variance_nrmse'],['Endpoint standardized SW₂','Ergodic variance NRMSE']):
  for m in colors:
   q=[r for r in rows if r['method']==m and r['valid_full_configuration']]
   ax.scatter([r[cost+'_seconds'] for r in q],[r[key] for r in q],s=36,color=colors[m],label=m.upper(),zorder=3)
   # Descriptive heldout grid minimum only; no interpolation used for time-to-target.
   best=float('inf');front=[]
   for r in sorted(q,key=lambda r:r[cost+'_seconds']):
    if r[key]<best:front.append(r);best=r[key]
   ax.plot([r[cost+'_seconds'] for r in front],[r[key] for r in front],color=colors[m],alpha=.8)
   for r in q:
    if r['iteration']==2048:
     ci=r[key+'_ci95'];ax.errorbar(r[cost+'_seconds'],r[key],yerr=[[r[key]-ci[0]],[ci[1]-r[key]]],color=colors[m],capsize=2,alpha=.5,fmt='none')
  ax.set(xscale='log',yscale='log',xlabel=f'Setup-inclusive {cost} time (T4 seconds)',ylabel=label);ax.grid(alpha=.2);ax.legend(fontsize=9)
 axes[0].axhline(.3,color='#777777',ls=':',lw=1);axes[1].axhline(.5,color='#777777',ls=':',lw=1)
 fig.suptitle('P3A held-out: fixed configurations, 8 seeds ×512 replicas')
 fig.text(.5,.01,'Descriptive frontier; any failed seed excludes the complete configuration. Error bars: pointwise 95% seed bootstrap at K2048.',ha='center',fontsize=8)
 fig.tight_layout(rect=[0,.04,1,.95]);fig.savefig(OUT/f'frontier_{cost}.png',dpi=180);fig.savefig(OUT/f'frontier_{cost}.pdf');plt.close(fig)
last=[r for r in rows if r['iteration']==2048];fig,axes=plt.subplots(1,2,figsize=(11.5,4.4))
for m in ['cv1','cv2']:
 q=[r for r in last if r['method']==m and r['valid_full_configuration']]
 axes[0].plot([r['batch'] for r in q],[r['variance_nrmse'] for r in q],marker='o',color=colors[m],label=m.upper())
 for r in q:axes[0].annotate(f"h={r['h']}",(r['batch'],r['variance_nrmse']),xytext=(0,8),textcoords='offset points',fontsize=7,ha='center')
axes[0].set(xscale='log',yscale='log',xlabel='Minibatch size (selected h varies)',ylabel='Variance NRMSE at K2048',ylim=(.16,60));axes[0].axhline(.5,color='#777',ls=':');axes[0].legend();axes[0].grid(alpha=.2)
labels=[r['method'].upper()+'\nn='+str(r['batch']) for r in last];axes[1].bar(np.arange(len(last)),[r['failed_runs']/8 for r in last],color=[colors[r['method']] for r in last]);axes[1].set_xticks(np.arange(len(last)),labels,fontsize=7);axes[1].set(ylabel='Fraction of runs with ≥1 failed replica',ylim=(0,.6));axes[1].grid(axis='y',alpha=.2)
fig.suptitle('P3A held-out accuracy and stability at K2048');fig.text(.5,.01,'CV2 n32 failed selection and was not run. Failed configurations have no pooled accuracy estimate.',ha='center',fontsize=8);fig.tight_layout(rect=[0,.05,1,.95]);fig.savefig(OUT/'batch_stability.png',dpi=180);fig.savefig(OUT/'batch_stability.pdf');plt.close(fig)
# Independent posterior-location MSE audit from the unchanged pilot.
pilot=json.loads(Path('results/p3a_bank_t4_pilot_v1/gradient_mse.json').read_text());fig,ax=plt.subplots(figsize=(6.2,4.2))
for m,color in [('sgld','#777777'),('cv1',colors['cv1']),('cv2',colors['cv2'])]:
 q=[r for r in pilot if r['method']==m];ax.plot([r['batch'] for r in q],[r['whitened_mse_mean'] for r in q],marker='o',color=color,label=m.upper())
ax.set(xscale='log',yscale='log',xlabel='Minibatch size',ylabel='Mean squared whitened gradient error');ax.legend();ax.grid(alpha=.2);ax.set_title('Independent posterior-location gradient audit (pilot)');fig.text(.5,.01,'24 common locations ×32 paired batches; diagnostic means, no scaling-rate inference.',ha='center',fontsize=8);fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(OUT/'gradient_mse.png',dpi=180);fig.savefig(OUT/'gradient_mse.pdf');plt.close(fig)

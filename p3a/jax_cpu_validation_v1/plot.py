from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'exports/p3a_jax_cpu_validation_review_v1';a=json.loads((OUT/'analysis.json').read_text());rows=a['configuration_checkpoints'];colors={'cv1':'#1769aa','cv2':'#e2711d','ula':'#333333'}
for cost in ['warm','cold']:
 fig,axes=plt.subplots(1,2,figsize=(11.5,4.6))
 for ax,key,label in zip(axes,['sw2','variance_nrmse'],['Endpoint standardized SW₂','Ergodic variance NRMSE']):
  for m in colors:
   q=[r for r in rows if r['method']==m and r['valid_full_configuration']];ax.scatter([r[cost+'_seconds'] for r in q],[r[key] for r in q],s=25,color=colors[m],alpha=.65,label=m.upper(),zorder=3);best=float('inf');front=[]
   for r in sorted(q,key=lambda r:r[cost+'_seconds']):
    if r[key]<best:front.append(r);best=r[key]
   ax.plot([r[cost+'_seconds'] for r in front],[r[key] for r in front],color=colors[m],lw=1.7)
   for r in front:
    interval=r[key+'_ci95'];ax.errorbar(r[cost+'_seconds'],r[key],yerr=[[r[key]-interval[0]],[interval[1]-r[key]]],color=colors[m],capsize=2,alpha=.4,fmt='none')
  ax.set(xscale='log',yscale='log',xlabel=f'Setup-inclusive {cost} time (JAX CPU seconds)',ylabel=label);ax.grid(alpha=.2);ax.legend(fontsize=9)
 axes[0].axhline(.3,color='#777',ls=':',lw=1);axes[1].axhline(.5,color='#777',ls=':',lw=1)
 fig.suptitle('P3A amended search: independent validation, 8 seeds ×512 replicas')
 fig.text(.5,.012,'Descriptive grid frontier; failed full configurations excluded. Pointwise 95% seed bootstrap on frontier points. ULA stops at K2048.',ha='center',fontsize=8)
 fig.tight_layout(rect=[0,.055,1,.95]);fig.savefig(OUT/f'frontier_{cost}.png',dpi=180);fig.savefig(OUT/f'frontier_{cost}.pdf');plt.close(fig)
# Explicitly show validation search coverage and failure rates; blank cells are untested candidates.
steps=[.0015625,.003125,.0046875,.00625,.009375,.0125];batches=[1,4,16,32,64,256];last=[r for r in rows if r['iteration']==r['horizon']];fig,axes=plt.subplots(1,2,figsize=(11.5,4.7));cmap=plt.cm.Oranges.copy();cmap.set_bad('#d9d9d9')
for ax,m in zip(axes,['cv1','cv2']):
 v=np.full((6,6),np.nan)
 for r in last:
  if r['method']==m:v[batches.index(r['batch']),steps.index(r['h'])]=r['failed_runs']/8
 im=ax.imshow(np.ma.masked_invalid(v),vmin=0,vmax=1,cmap=cmap,aspect='auto')
 for i in range(6):
  for j in range(6):ax.text(j,i,'—' if np.isnan(v[i,j]) else f'{int(round(v[i,j]*8))}/8',ha='center',va='center',fontsize=9,color='white' if v[i,j]>.65 else 'black')
 ax.set_xticks(range(6),[str(h) for h in steps],rotation=35,ha='right',fontsize=8);ax.set_yticks(range(6),[str(n) for n in batches]);ax.set(xlabel='Pilot-selected step size',ylabel='Minibatch size',title=m.upper())
fig.colorbar(im,ax=axes,fraction=.025,pad=.025,label='Fraction of validation runs with ≥1 failed replica');fig.suptitle('Full-horizon failures among the 34 fixed validation candidates');fig.text(.5,.015,'Gray: not included in validation. Main horizon K8192; all failed runs and replicas retained.',ha='center',fontsize=8);fig.subplots_adjust(left=.065,right=.88,bottom=.25,top=.83,wspace=.25);fig.savefig(OUT/'failure_grid.png',dpi=180);fig.savefig(OUT/'failure_grid.pdf');plt.close(fig)
# Two distinct outputs: fixed primary decisions versus post-validation descriptive minima.
fig,ax=plt.subplots(figsize=(7,4.5))
for m in ['cv1','cv2','ula']:
 q=[r for r in a['exploratory_grid_minima'] if r['mode']=='joint' and r['cost']=='warm' and r['method']==m and r['choice']];ax.plot([r['threshold'] for r in q],[r['choice']['warm_seconds'] for r in q],marker='o',color=colors[m],label=m.upper())
ax.set(yscale='log',xlabel='Variance NRMSE threshold (also SW₂≤0.3)',ylabel='Observed minimum JAX CPU warm time (seconds)');ax.set_xticks([.5,.4,.3,.2]);ax.invert_xaxis();ax.grid(alpha=.2);ax.legend();ax.set_title('Descriptive accuracy sensitivity within the fixed validation grid');fig.text(.5,.01,'Selected after validation; potentially optimistic. CV2 does not reach0.2; these are not primary fixed-choice speedups.',ha='center',fontsize=8);fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(OUT/'descriptive_threshold_times.png',dpi=180);fig.savefig(OUT/'descriptive_threshold_times.pdf');plt.close(fig)

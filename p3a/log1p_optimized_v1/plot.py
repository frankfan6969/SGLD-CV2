from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[2];O=R/'exports/p3a_log1p_optimized_v1';a=json.loads((O/'analysis.json').read_text());rows=[q for q in a['configuration_checkpoints'] if q['eligible']];fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for i,cost in enumerate(['warm','cold']):
 for j,(metric,label) in enumerate([('sw2','Endpoint standardized SW2'),('variance_nrmse','Ergodic variance NRMSE')]):
  x=ax[i,j]
  for method,color in [('cv1','tab:blue'),('cv2','tab:orange')]:
   q=sorted([q for q in rows if q['method']==method],key=lambda q:(q[cost+'_seconds'],q[metric]));x.scatter([q[cost+'_seconds'] for q in q],[q[metric] for q in q],alpha=.35,s=20,color=color,label=method.upper());front=[];best=np.inf
   for row in q:
    if row[metric]<best:front.append(row);best=row[metric]
   if front:x.plot([q[cost+'_seconds'] for q in front],[q[metric] for q in front],'o-',ms=3,color=color)
  x.axhline(.3 if metric=='sw2' else .5,color='gray',ls=':',lw=1);x.set(xscale='log',yscale='log',xlabel=f'Setup-inclusive {cost} CPU seconds',ylabel=label);x.grid(alpha=.2);x.legend(fontsize=9)
fig.suptitle('Log1p Bank: independent validation; marginal2D frontiers over frozen candidate grid',fontsize=13);fig.supxlabel('All eligible points shown; any failure excludes full configuration. Joint3D frontiers in JSON.',fontsize=9);fig.savefig(O/'accuracy_cost_frontiers.png',dpi=180);fig.savefig(O/'accuracy_cost_frontiers.pdf');plt.close(fig)
fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for i,cost in enumerate(['warm','cold']):
 sweep=sorted([q for q in a['threshold_sweep'] if q['cost']==cost],key=lambda q:q['variance_threshold']);xx=[q['variance_threshold'] for q in sweep]
 for m,col in [('cv1','tab:blue'),('cv2','tab:orange')]:
  for typ,ls in [('point_choice','-'),('simultaneous_supported_choice','--')]:
   yy=[q['methods'][m][typ][cost+'_seconds'] if q['methods'][m][typ] else np.nan for q in sweep];ax[i,0].plot(xx,yy,ls,color=col,label=m.upper()+(' point' if typ=='point_choice' else ' simultaneous-supported'),marker='o',ms=2)
 point=np.array([q['point_speedup_cv1_over_cv2'] if q['point_speedup_cv1_over_cv2'] is not None else np.nan for q in sweep]);lo=np.array([q['conditional_speedup_ci95'][0] if q['point_speedup_cv1_over_cv2'] is not None else np.nan for q in sweep]);hi=np.array([q['conditional_speedup_ci95'][1] if q['point_speedup_cv1_over_cv2'] is not None else np.nan for q in sweep]);ax[i,1].plot(xx,point,'o-',ms=3,color='tab:purple');ax[i,1].fill_between(xx,lo,hi,color='tab:purple',alpha=.15);ax[i,1].axhline(1,c='k',ls='--',lw=1)
 for x in ax[i]:x.set_xlim(.605,.195);x.set_xlabel('Variance NRMSE target (stricter →), SW2≤0.3');x.grid(alpha=.2)
 ax[i,0].set(yscale='log',ylabel=f'Minimum grid {cost} seconds');ax[i,0].legend(fontsize=8);ax[i,1].set_ylabel(f'CV1/CV2 {cost} time ratio (>1 favors CV2)')
fig.suptitle('Optimized-grid threshold sensitivity: points vs simultaneous accuracy support',fontsize=13);fig.supxlabel('Pointwise paired bootstrap ratio intervals, conditional on both attaining; missing values remain Not Reached.',fontsize=9);fig.savefig(O/'optimized_thresholds.png',dpi=180);fig.savefig(O/'optimized_thresholds.pdf');plt.close(fig)
# Failure map includes every frozen validated candidate, not only successful points.
fig,ax=plt.subplots(figsize=(10,4),layout='constrained');qs=[q for q in a['configuration_checkpoints'] if q['K']==max(r['K'] for r in a['configuration_checkpoints'])]
for m,col,marker in [('cv1','tab:blue','o'),('cv2','tab:orange','s')]:
 q=[q for q in qs if q['method']==m];ax.scatter([q['batch'] for q in q],[q['h'] for q in q],c=[q['failed_replicas'] for q in q],cmap='Reds',marker=marker,s=85,edgecolors=col,label=m.upper())
 for row in q:ax.annotate(str(row['failed_replicas']),(row['batch'],row['h']),xytext=(5,3),textcoords='offset points',fontsize=7)
ax.set(xscale='log',yscale='log',xlabel='Batch size',ylabel='Frozen h',title='Independent-validation candidates: labels are failed replicas/4096');ax.legend();ax.grid(alpha=.2);fig.savefig(O/'candidate_failures.png',dpi=180);fig.savefig(O/'candidate_failures.pdf');plt.close(fig)

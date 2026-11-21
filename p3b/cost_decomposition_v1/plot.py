from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[2];O=R/'exports/p3b_cost_decomposition_v1';a=json.loads((O/'analysis.json').read_text());Ns=[2000,8000,32000,128000];thresholds=[.5,.4,.3,.2];fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for x,t in zip(ax.ravel(),thresholds):
 for method,offset,col in [('cv1',-.19,'tab:blue'),('cv2',.19,'tab:orange')]:
  q=sorted([q for q in a['optimal_configurations'] if q['cost']=='warm' and q['choice_type']=='point_choice' and q['threshold']==t and q['method']==method],key=lambda q:q['N']);pos=np.arange(4)+offset;anchor=np.array([r['anchor_seconds'] for r in q]);setup=np.array([r['setup_seconds'] for r in q]);sampler=np.array([r['prefix_sampling_seconds'] for r in q]);x.bar(pos,anchor,.35,color='gray',label='Mode attribution' if method=='cv1' else None);x.bar(pos,setup,.35,bottom=anchor,color='goldenrod',label='Prepare / state setup' if method=='cv1' else None);x.bar(pos,sampler,.35,bottom=anchor+setup,color=col,label=method.upper()+' sampling')
 x.set(title=f'Variance NRMSE≤{t:.2f}, SW2≤0.3',xticks=np.arange(4),xticklabels=['2k','8k','32k','128k'],xlabel='N; left CV1, right CV2',ylabel='Setup-inclusive warm seconds');x.grid(axis='y',alpha=.2)
ax[0,0].legend(fontsize=8);fig.suptitle('Saved optimal-prefix cost decomposition: common setup increasingly dilutes speedup');fig.savefig(O/'cost_components.png',dpi=180);fig.savefig(O/'cost_components.pdf');plt.close(fig)
fig,ax=plt.subplots(2,2,figsize=(11,7),layout='constrained')
for x,t in zip(ax.ravel(),thresholds):
 q=sorted([q for q in a['speedup_decomposition'] if q['cost']=='warm' and q['choice_type']=='point_choice' and q['threshold']==t],key=lambda q:q['N']);x.plot(Ns,[r['warm_speedup'] for r in q],'o-',label='Actual setup-inclusive');x.plot(Ns,[r['sampling_only_speedup'] for r in q],'s--',label='Sampling-only accounting');x.plot(Ns,[r['without_anchor_attribution_speedup'] for r in q],'^:',label='Remove mode attribution only');x.set(xscale='log',title=f'NRMSE≤{t:.2f}',xlabel='N',ylabel='CV1/CV2 time ratio');x.axhline(1,c='k',ls=':',lw=1);x.grid(alpha=.2)
ax[0,0].legend(fontsize=8);fig.suptitle('Where the speedup decline comes from; accounting views are not new official rankings');fig.savefig(O/'speedup_accounting.png',dpi=180);fig.savefig(O/'speedup_accounting.pdf');plt.close(fig)
q=[q for q in a['matched_configuration_controls'] if q['eligible']];keys=set.intersection(*[{(r['batch'],r['h_canonical']) for r in q if r['N']==N} for N in Ns]);fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained')
for n,h in sorted(keys):
 rr=sorted([r for r in q if (r['batch'],r['h_canonical'])==(n,h)],key=lambda r:r['N']);v=np.array([r['cv2_over_cv1_step_ratio'] for r in rr]);lo=np.array([r['ratio_ci95'][0] for r in rr]);hi=np.array([r['ratio_ci95'][1] for r in rr]);ax.errorbar(Ns,v,yerr=[v-lo,hi-v],marker='o',capsize=2,label=f'n={n},h={h:g}')
ax.set(xscale='log',xlabel='N',ylabel='CV2/CV1 matched sampling time per batched step',title='Same n,h,K=4096,R=512; includes RNG, excludes setup/compilation');ax.axhline(1,c='k',ls='--',lw=1);ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2);fig.savefig(O/'matched_per_step.png',dpi=180);fig.savefig(O/'matched_per_step.pdf');plt.close(fig)

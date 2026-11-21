from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[2];O=R/'exports/p3a_log1p_checkpoint_v1';a=json.loads((O/'analysis.json').read_text());fig,ax=plt.subplots(1,3,figsize=(13,4),layout='constrained')
for m,col in [('cv1','tab:blue'),('cv2','tab:orange')]:
 r=[q for q in a['checkpoints'] if q['method']==m and q['eligible']];ks=[q['K'] for q in r]
 for x,key,label in zip(ax,['sw2','variance_nrmse','warm_seconds'],['Endpoint standardized SW2','Variance NRMSE','Setup-inclusive warm seconds']):
  val=np.array([q[key] for q in r]);ci=np.array([q['warm_ci95' if key=='warm_seconds' else key+'_ci95'] for q in r]);x.plot(ks,val,'o-',color=col,label=m.upper());x.fill_between(ks,ci[:,0],ci[:,1],color=col,alpha=.15);x.set(xlabel='Iterations K',ylabel=label,xticks=[128,256,384,512]);x.grid(alpha=.2)
ax[0].axhline(.3,c='k',ls='--',lw=1);ax[1].axhline(.5,c='k',ls='--',lw=1);ax[0].legend();fig.suptitle('Log1p Bank: fixed n=16,h=0.025;8 paired seeds ×512 replicas; pointwise95% intervals');fig.savefig(O/'checkpoint_accuracy.png',dpi=180);fig.savefig(O/'checkpoint_accuracy.pdf')

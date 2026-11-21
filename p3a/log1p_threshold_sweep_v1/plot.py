from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[2];O=R/'exports/p3a_log1p_threshold_sweep_v1';rows=json.loads((O/'analysis.json').read_text())['sweep'];x=np.array([q['variance_threshold'] for q in rows]);fig,ax=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
for m,col in [('cv1','tab:blue'),('cv2','tab:orange')]:
 y=np.array([q[m+'_seconds'] if q[m+'_seconds'] is not None else np.nan for q in rows]);ax[0].plot(x,y,'o-',ms=3,label=m.upper(),color=col)
y=np.array([q['point_speedup_cv1_over_cv2'] if q['point_speedup_cv1_over_cv2'] is not None else np.nan for q in rows]);lower=np.array([q['conditional_speedup_ci95'][0] if q['point_speedup_cv1_over_cv2'] is not None else np.nan for q in rows]);upper=np.array([q['conditional_speedup_ci95'][1] if q['point_speedup_cv1_over_cv2'] is not None else np.nan for q in rows]);ax[1].plot(x,y,'o-',ms=3,color='tab:purple');ax[1].fill_between(x,lower,upper,color='tab:purple',alpha=.18,label='Pointwise95% paired bootstrap');ax[1].axhline(1,c='k',ls='--',lw=1)
for a in ax:
 a.set_xlim(.605,.195);a.set_xlabel('Variance NRMSE threshold (stricter →)');a.grid(alpha=.2);a.axvline(.5,c='gray',ls=':',lw=1)
 for q in rows:
  if q['status']=='CV2_only_reached':a.axvspan(q['variance_threshold']-.005,q['variance_threshold']+.005,color='tab:orange',alpha=.08,lw=0)
ax[0].set_ylabel('First tested-grid attainment: warm seconds');ax[0].legend();ax[1].set_ylabel('CV1 time / CV2 time (>1 favors CV2)');ax[1].legend(fontsize=8);fig.suptitle('Log1p Bank: full exploratory threshold sweep at fixed n=16,h=0.025',fontsize=13);fig.supxlabel('Orange shading: only CV2 reaches by512. Below0.37 neither reaches. No finite ratio for Not Reached.',fontsize=9);fig.savefig(O/'threshold_sensitivity.png',dpi=180);fig.savefig(O/'threshold_sensitivity.pdf')

"""Post-replay exploratory single-step counterfactual; not a new trajectory."""
from pathlib import Path
import numpy as np,json
from scipy.special import expit
from threadpoolctl import threadpool_limits
R=Path(__file__).resolve().parents[2];A=R/'results/p3a_bank_t4_heldout_v1/frozen_bank_assets';O=R/'exports/p3a_mechanism_diagnostic_v1'
with threadpool_limits(limits=1):
 a=np.load(A/'target.npz');X,y,anchor=a['X'],a['y'],a['anchor'];P=np.linalg.inv(a['H']+1e-10*np.eye(X.shape[1]));L=np.linalg.cholesky(P);W=a['W'];p0=expit(X@anchor);w=p0*(1-p0);Hd=(X.T*w)@X;ga=X.T@(p0-y);records=[]
 for r in json.loads((O/'failure_replay.json').read_text()):
  z=np.load(O/f"replay_n{r['batch']}_h{r['h']}_seed{r['seed']}.npz");t=z['theta_before'][-1];idx=z['indices'][-1];delta=t-anchor;rr1=expit(X[idx]@t)-p0[idx];rr2=rr1-w[idx]*(X[idx]@delta);s={}
  for m,rr in [('cv1',rr1),('cv2',rr2)]:
   g=ga+(Hd@delta if m=='cv2' else 0)+(len(X)/len(idx))*X[idx].T@rr+t;proposal=t-r['h']*(P@g)+np.sqrt(2*r['h'])*(L@z['normal'][-1]);s[m]=dict(counterfactual_radius_same_theta_batch_noise=float(np.linalg.norm((proposal-anchor)@W)),sampled_residual_for_29182=float(rr[np.flatnonzero(idx==29182)[0]]))
  row=X[29182];s['observation_29182']=dict(anchor_logit=float(row@anchor),delta_logit=float(row@delta),anchor_weight=float(w[29182]),label=float(y[29182]),design_mahalanobis=float(row@P@row),largest_standardized_numeric_design_abs=float(abs(row).max()),largest_feature='previous',selection='Identified after replay; no observation removed or target changed')
  records.append(dict(batch=r['batch'],h=r['h'],seed=r['seed'],scope='Post-replay exploratory fixed-input single-step counterfactual; not a CV1 trajectory',**s))
 (O/'trigger_counterfactual.json').write_text(json.dumps(records,indent=2)+'\n')

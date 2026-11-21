from pathlib import Path
import json,os,hashlib,numpy as np
R=Path(__file__).resolve().parents[2];D=R/'results/p5_baseline_extension_v1';E=R/'exports/p5_baseline_extension_v1';H=Path(__file__).parent

def load(p):return json.loads(Path(p).read_text())
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(v,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n');os.replace(tmp,p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def score(endpoints,sums,k,anchor,W,projections,refs):
 if not np.isfinite(endpoints).all():return dict(status='failed_ensemble',references=None)
 d=len(anchor);avg=sums/k;var=avg[:,d:]-avg[:,:d]**2;xp=np.sort(((endpoints-anchor)@W)@projections,axis=0);values=[]
 for ref in refs:
  q=(ref-anchor)@W;yp=np.sort(q@projections,axis=0);size=np.lcm(len(xp),len(yp));sw=float(np.sqrt(np.mean((np.repeat(xp,size//len(xp),axis=0)-np.repeat(yp,size//len(yp),axis=0))**2)));v=q.var(0);rmse=float(np.sqrt(np.mean(((var-v)/v)**2,axis=0)).mean());values.append(dict(sw2=sw,variance_nrmse=rmse))
 return dict(status='ok',references=values)
def arrays(path,**kw):
 path=Path(path);tmp=path.with_name(path.name+'.tmp')
 with tmp.open('wb') as f:np.savez_compressed(f,**kw)
 os.replace(tmp,path)
def ranks(points):
 p=np.asarray(points);left=np.arange(len(p));out=np.full(len(p),-1);rank=0
 while len(left):
  q=p[left];dom=np.all(q[:,None]<=q[None,:],axis=2)&np.any(q[:,None]<q[None,:],axis=2);front=~dom.any(0);out[left[front]]=rank;left=left[~front];rank+=1
 return out

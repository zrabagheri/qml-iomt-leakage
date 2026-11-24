"""Outer temporal 5-fold evaluation of the configuration chosen on the inner split
(L=6, 15 epochs, lr=0.02, 3-seed ensemble) vs the paper's config (L=4, lr=0.05)."""
import numpy as np, json
from prep import *; from vqc import VQC; from train import train_central
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score
X,y,cat,const=load(); S=sets(X,const); A=X[S['clean']].values
out={}
for name,(L,lr) in {'paper':(4,0.05),'new':(6,0.02)}.items():
    single,ens,pr,f1,ba=[],[],[],[],[]
    for i,trb,te in folds(X):
        tr=np.concatenate(trb); P=Pipe(4,True).fit(A[tr]); Xtr=P.transform(A[tr]); Xte=P.transform(A[te])
        ps=[]
        for s in [0,1,2]:
            m=VQC(4,L,s); train_central(m,Xtr,y[tr],15,128,lr,s); ps.append(m.predict_proba(Xte))
        single.append(np.mean([roc_auc_score(y[te],p) for p in ps])); pe=np.mean(ps,0)
        ens.append(roc_auc_score(y[te],pe)); pr.append(average_precision_score(y[te],pe))
        f1.append(f1_score(y[te],pe>=0.5)); ba.append(balanced_accuracy_score(y[te],pe>=0.5))
    out[name]=dict(single=(np.mean(single),np.std(single)),ens=(np.mean(ens),np.std(ens)),prauc=np.mean(pr),f1=np.mean(f1),bacc=np.mean(ba),folds_ens=ens)
    print(name,{k:(np.round(v,3).tolist() if not isinstance(v,list) else np.round(v,3).tolist()) for k,v in out[name].items()},flush=True)
json.dump({k:{kk:(list(map(float,np.ravel(vv)))) for kk,vv in v.items()} for k,v in out.items()},open('../results/tune_broad_outer.json','w'),indent=1)

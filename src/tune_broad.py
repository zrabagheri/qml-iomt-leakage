"""Honest attempt to improve the 4-qubit VQC. Selection ONLY on the inner validation split
(blocks 1-3 train, block 4 validation), as in the paper. Outer test folds untouched here."""
import numpy as np, time, itertools, json, sys
from prep import *; from vqc import VQC; from train import train_central
from sklearn.metrics import roc_auc_score
X,y,cat,const=load(); S=sets(X,const); A=X[S['clean']].values
fl=list(folds(X))
# inner split: blocks are the 5 temporal blocks; reconstruct block indices
blocks=[fl[i][2] for i in range(5)]
tr=np.concatenate(blocks[0:3]); va=blocks[3]
P=Pipe(4,True).fit(A[tr]); Xtr=P.transform(A[tr]); Xva=P.transform(A[va])
res=[]
for L,ep,lr in itertools.product([4,6,8],[15,30],[0.05,0.02]):
    preds=[]
    t=time.time()
    for s in [0,1,2]:
        m=VQC(4,L,s); train_central(m,Xtr,y[tr],ep,128,lr,s); preds.append(m.predict_proba(Xva))
    single=np.mean([roc_auc_score(y[va],p) for p in preds]); ens=roc_auc_score(y[va],np.mean(preds,0))
    res.append(dict(L=L,ep=ep,lr=lr,single=single,ens=ens)); print(res[-1],'%.0fs'%(time.time()-t),flush=True)
json.dump(res,open('../results/tune_broad_inner.json','w'),indent=1)

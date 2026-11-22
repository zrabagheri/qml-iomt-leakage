import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from prep import *; from vqc import VQC, LinearModel; from train import train_fedavg, epsilon
from exp_main import metrics
X,y,cat,const=load(); S=sets(X,const); A=X[S['clean']].values
rows=[]
cfg=[('VQC',1.0,0.0),('VQC',5.0,1.0),('VQC',5.0,0.0),('LinFull',1.0,0.0)]
for i,trb,te in folds(X):
    tr=np.concatenate(trb)
    for name,C,sig in cfg:
        P=Pipe(4,True).fit(A[tr]) if name=='VQC' else Pipe().fit(A[tr])
        clients=[(P.transform(A[b]),y[b]) for b in trb]; Xte=P.transform(A[te])
        for s in [0,1,2]:
            m=VQC(4,4,s) if name=='VQC' else LinearModel(len(S['clean']),s)
            info=train_fedavg(m,clients,15,1,128,0.05 if name=='VQC' else 0.02,s,clip=C,sigma=sig)
            eps=max(epsilon(c['q'],sig,c['steps']) for c in info) if sig>0 else float('inf')
            rows.append(dict(model=name,clip=C,sigma=sig,eps=eps,fold=i,seed=s,**metrics(y[te],m.predict_proba(Xte))))
    print('fold',i,flush=True)
pd.DataFrame(rows).to_csv('res_E3b.csv',index=False)
f=pd.DataFrame(rows).groupby(['model','clip','sigma','fold']).auc.mean().reset_index()
print(f.groupby(['model','clip','sigma']).auc.agg(['mean','std']).round(3))

import numpy as np, itertools, json
from prep import *; from vqc import VQC; from train import train_central
from sklearn.metrics import roc_auc_score
X,y,cat,const=load(); S=sets(X,const); A=X[S['clean']].values
o=np.argsort(X.Packet_num.values); B=np.array_split(o,5)
tr=np.concatenate(B[:3]); va=B[3]          # blocks 1-3 train, block 4 validation, block 5 never touched
out=[]
for q,L,lr in itertools.product([4,6,8],[2,4],[0.02,0.05]):
    P=Pipe(q,True).fit(A[tr]); Xtr,Xva=P.transform(A[tr]),P.transform(A[va])
    m=VQC(q,L,0); train_central(m,Xtr,y[tr],15,128,lr,0)
    a=roc_auc_score(y[va],m.predict_proba(Xva)); out.append((q,L,lr,a)); print(q,L,lr,round(a,3),flush=True)
json.dump(out,open('tune.json','w'))

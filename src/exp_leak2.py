import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from prep import *
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
X,y,cat,const=load(); S=sets(X,const)
S['idonly']=['SrcMac','Flgs','Sport','Packet_num']; S['mac']=['SrcMac']
R=[]
for fs in ['all','idonly','mac','no_id','clean','network','bio']:
    A=X[S[fs]].values
    for proto in ['random','temporal']:
        sp=[(np.concatenate(trb),te) for i,trb,te in folds(X)] if proto=='temporal' else list(StratifiedKFold(5,shuffle=True,random_state=0).split(A,y))
        for mname in ['LR','RF']:
            a=[]
            for tr,te in sp:
                P=Pipe().fit(A[tr]); m=LogisticRegression(max_iter=3000,class_weight='balanced') if mname=='LR' else RandomForestClassifier(300,class_weight='balanced',n_jobs=-1,random_state=0)
                m.fit(P.transform(A[tr]),y[tr]); a.append(roc_auc_score(y[te],m.predict_proba(P.transform(A[te]))[:,1]))
            R.append(dict(fs=fs,nf=len(S[fs]),proto=proto,model=mname,auc=np.mean(a),sd=np.std(a,ddof=1)))
    print(fs,flush=True)
pd.DataFrame(R).to_csv('res_leak2.csv',index=False)
print(pd.DataFrame(R).pivot_table(index=['fs','nf'],columns=['model','proto'],values='auc').round(3))

import pandas as pd, numpy as np, json
from prep import *
X,y,cat,const=load(); S=sets(X,const)
N={}
N['n']=len(y); N['npos']=int(y.sum()); N['prev']=round(100*y.mean(),1)
N['nspoof']=int((cat=='Spoofing').sum()); N['nalt']=int((cat=='Data Alteration').sum())
N['nconst']=len(const); N['nclean']=len(S['clean']); N['nnet']=len(S['network']); N['nbio']=len(S['bio'])
o=np.argsort(X.Packet_num.values); pr=[100*y[b].mean() for b in np.array_split(o,5)]
N['prevrange']=f'{min(pr):.1f}–{max(pr):.1f}'
def fs(df,keys,cols=('auc','prauc','f1','bacc')):
    f=df.groupby(list(keys)+['fold'])[list(cols)].mean().reset_index()
    return f.groupby(list(keys))[list(cols)].agg(['mean','std'])
E1=fs(pd.read_csv('res_E1.csv'),['model','feats','q'])
E2=fs(pd.read_csv('res_E2.csv'),['model','feats','protocol'])
e3=pd.read_csv('res_E3.csv'); E3=fs(e3,['model','sigma'])
EPS=e3.groupby('sigma').eps.max().to_dict()
E3b=fs(pd.read_csv('res_E3b.csv'),['model','clip','sigma'])
L2=pd.read_csv('res_leak2.csv').set_index(['fs','model','proto'])
json.dump({k:(v if not isinstance(v,np.generic) else v.item()) for k,v in N.items()},open('numbers.json','w'),ensure_ascii=False)

import pandas as pd, numpy as np
ID=['Dir','SrcAddr','DstAddr','Sport','Dport','SrcMac','DstMac','Packet_num']
ART=['Flgs']
BIO=['Temp','SpO2','Pulse_Rate','SYS','DIA','Heart_rate','Resp_Rate','ST']
def load():
    d=pd.read_csv(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', 'data', 'wustl.csv') if not __import__('os').path.exists('data/wustl.csv') else 'data/wustl.csv')
    d['Sport']=pd.to_numeric(d['Sport'],errors='coerce').fillna(-1)
    for c in ['Dir','SrcAddr','DstAddr','SrcMac','DstMac','Flgs']:
        d[c]=d[c].astype('category').cat.codes
    y=d.Label.values; cat=d['Attack Category'].values
    X=d.drop(columns=['Label','Attack Category'])
    const=[c for c in X if X[c].nunique()<=1]
    return X,y,cat,const
def sets(X,const):
    allf=[c for c in X if c not in const]
    nid=[c for c in allf if c not in ID]
    clean=[c for c in nid if c not in ART]
    net=[c for c in clean if c not in BIO]
    return {'all':allf,'no_id':nid,'clean':clean,'network':net,'bio':[c for c in BIO if c in clean]}

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
def slog(A): return np.sign(A)*np.log1p(np.abs(A))
def folds(X,k=5):
    o=np.argsort(X.Packet_num.values); B=np.array_split(o,k)
    for i in range(k): yield i,[B[j] for j in range(k) if j!=i],B[i]
class Pipe:
    """signed-log -> standardise -> (optional) PCA -> (optional) angle map; fitted on training data only."""
    def __init__(self,ncomp=None,angles=False): self.n=ncomp; self.ang=angles
    def fit(self,A):
        A=slog(A); self.sc=StandardScaler().fit(A); Z=self.sc.transform(A)
        if self.n: self.pca=PCA(self.n,random_state=0).fit(Z); Z=self.pca.transform(Z); self.s2=Z.std(0)+1e-9
        return self
    def transform(self,A):
        Z=self.sc.transform(slog(A))
        if self.n: Z=self.pca.transform(Z)/self.s2
        return 2*np.arctan(Z) if self.ang else Z

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, pandas as pd, numpy as np
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.5})
d=pd.read_csv('res_E3.csv'); f=d.groupby(['model','sigma','fold']).auc.mean().reset_index()
g=f.groupby(['model','sigma']).auc.agg(['mean','std'])
b=pd.read_csv('res_E3b.csv'); fb=b.groupby(['model','clip','sigma','fold']).auc.mean().reset_index().groupby(['model','clip','sigma']).auc.agg(['mean','std'])
sig=[0.0,0.6,1.0,1.5]; xl=['No DP','ε=20.5','ε=6.4','ε=3.2']; x=np.arange(4)
fig,ax=plt.subplots(figsize=(3.5,1.9))
for m,lab,c,mk in [('VQC','VQC (4 qubits), C=1','#DD8452','o'),('LinPCA','LR-PCA4, C=1','#55A868','s'),('LinFull','LR-full, C=1','#4C72B0','^')]:
    mu=[g.loc[(m,s),'mean'] for s in sig]; sd=[g.loc[(m,s),'std'] for s in sig]
    ax.errorbar(x+{'VQC':0,'LinPCA':-0.06,'LinFull':0.06}[m],mu,yerr=sd,color=c,marker=mk,ms=3.5,lw=1,capsize=1.5,elinewidth=0.5,label=lab)
mu=fb.loc[('VQC',5.0,1.0),'mean']; sd=fb.loc[('VQC',5.0,1.0),'std']
ax.errorbar([2.12],[mu],yerr=[sd],color='#C44E52',marker='D',ms=3.5,capsize=1.5,elinewidth=0.5,ls='none',label='VQC, C=5')
ax.set_xticks(x); ax.set_xticklabels(xl); ax.set_ylabel('AUC (temporal CV)'); ax.set_ylim(0.66,0.9)
ax.legend(fontsize=5.8,frameon=False,ncol=2,loc='upper center')
for s in ['top','right']: ax.spines[s].set_visible(False)
plt.tight_layout(pad=0.2); plt.savefig('figs/Fig3_dp.png',dpi=600)

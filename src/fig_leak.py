import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, pandas as pd, numpy as np
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.5})
d=pd.read_csv('res_E2.csv')
f=d.groupby(['model','feats','protocol','fold']).auc.mean().reset_index()   # average seeds within fold
g=f.groupby(['model','feats','protocol']).auc.agg(['mean','std'])
conds=[('all','random'),('all','temporal'),('clean','random'),('clean','temporal')]
labels=['Identifiers\nrandom CV','Identifiers\ntemporal CV','Leak-free\nrandom CV','Leak-free\ntemporal CV']
x=np.arange(4); w=0.36
fig,ax=plt.subplots(figsize=(3.5,1.9))
for k,(m,c,lab) in enumerate([('RF','#4C72B0','Random forest'),('VQC','#DD8452','VQC (4 qubits)')]):
    mu=[g.loc[(m,a,b),'mean'] for a,b in conds]; sd=[g.loc[(m,a,b),'std'] for a,b in conds]
    bars=ax.bar(x+(k-.5)*w,mu,w,yerr=sd,color=c,label=lab,capsize=1.5,error_kw=dict(lw=0.6))
    for xi,v,e in zip(x+(k-.5)*w,mu,sd): ax.text(xi,v+e+0.012,f'{v:.2f}',ha='center',fontsize=5.5)
ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=6); ax.set_ylim(0.5,1.1); ax.set_ylabel('AUC')
ax.axhline(0.5,color='gray',lw=0.5); ax.legend(fontsize=6,loc='upper right',frameon=False,ncol=2,bbox_to_anchor=(1,1.13))
for s in ['top','right']: ax.spines[s].set_visible(False)
plt.tight_layout(pad=0.2); plt.savefig('figs/Fig2_leakage.png',dpi=600)
print(g.round(3))

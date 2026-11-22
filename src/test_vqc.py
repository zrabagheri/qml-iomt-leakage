import numpy as np
from vqc import forward, expval_and_grad
def RY(a): c,s=np.cos(a/2),np.sin(a/2); return np.array([[c,-s],[s,c]])
def kron_list(ms):
    o=np.array([[1.]]); 
    for m in ms: o=np.kron(o,m)
    return o
def cnot_full(q,c,t):
    n=2**q; U=np.zeros((n,n))
    for i in range(n):
        b=[(i>>(q-1-k))&1 for k in range(q)]
        if b[c]: b[t]^=1
        j=sum(bit<<(q-1-k) for k,bit in enumerate(b)); U[j,i]=1
    return U
def ref(x,theta,q,L):
    psi=np.zeros(2**q); psi[0]=1
    for l in range(L):
        psi=kron_list([RY(x[j]) for j in range(q)])@psi
        psi=kron_list([RY(theta[l,j]) for j in range(q)])@psi
        if q>1:
            for j in range(q): psi=cnot_full(q,j,(j+1)%q)@psi
    Z0=kron_list([np.diag([1,-1])]+[np.eye(2)]*(q-1))
    return psi@Z0@psi
r=np.random.default_rng(1)
for q,L in [(1,2),(2,2),(3,3),(4,2),(5,3)]:
    X=r.uniform(0,np.pi,(6,q)); th=r.uniform(-np.pi,np.pi,(L,q))
    a=forward(X,th,q,L); b=np.array([ref(x,th,q,L) for x in X])
    z,G=expval_and_grad(X,th,q,L); eps=1e-6; fd=np.zeros_like(G)
    for k in range(th.size):
        tp=th.ravel().copy(); tp[k]+=eps; tm=th.ravel().copy(); tm[k]-=eps
        fd[:,k]=(forward(X,tp.reshape(th.shape),q,L)-forward(X,tm.reshape(th.shape),q,L))/(2*eps)
    print(q,L,'max|sim-ref|=%.2e'%np.abs(a-b).max(),'max|grad-fd|=%.2e'%np.abs(G-fd).max())

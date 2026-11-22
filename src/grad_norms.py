"""Share of per-example gradient norms above the DP clipping bound C = 1 after centralised training
(seed 0), for every blocked temporal fold, computed on all training examples."""
import numpy as np
from prep import *
from vqc import VQC, LinearModel
from train import train_central
X, y, cat, const = load(); S = sets(X, const); A = X[S['clean']].values
res = {'VQC': [], 'LR-full': []}
for i, trb, te in folds(X):
    tr = np.concatenate(trb); yt = y[tr]; w = (yt == 0).sum() / (yt == 1).sum()
    Pa = Pipe(4, True).fit(A[tr]); Pf = Pipe().fit(A[tr]); Xa, Xf = Pa.transform(A[tr]), Pf.transform(A[tr])
    v = VQC(4, 4, 0); lm = LinearModel(len(S['clean']))
    train_central(v, Xa, yt, 15, 128, 0.05, 0); train_central(lm, Xf, yt, 15, 128, 0.02, 0)
    for name, (m, Z) in {'VQC': (v, Xa), 'LR-full': (lm, Xf)}.items():
        g, _ = m.per_example_grads(Z, yt, w); res[name].append((np.linalg.norm(g, axis=1) > 1).mean())
    print('fold', i + 1, {k: round(x[-1], 2) for k, x in res.items()}, flush=True)
for k, x in res.items(): print(f'{k}: share > C=1 mean {np.mean(x):.2f}, range {min(x):.2f}-{max(x):.2f}')

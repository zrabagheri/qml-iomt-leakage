"""Main experiments. Primary protocol: blocked temporal 5-fold CV on leak-free ('clean') features.
E1 centralised: classical (full features and PCA-q) vs VQC (q = 4, 6, 8).
E2 protocol sensitivity for the VQC and RF: random stratified split, and features with identifiers.
E3 federated (4 time-block clients) with/without example-level DP-SGD.
"""
import numpy as np, pandas as pd, sys, warnings, json
warnings.filterwarnings('ignore')
from prep import *
from vqc import VQC, LinearModel
from train import train_central, train_fedavg, epsilon
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, precision_score, balanced_accuracy_score

X, y, cat, const = load(); S = sets(X, const)
L_, LR_, EP, BS = 4, 0.05, 15, 128
SEEDS = [0, 1, 2]


def metrics(yt, p):
    t = (p >= 0.5).astype(int)
    return dict(auc=roc_auc_score(yt, p), prauc=average_precision_score(yt, p), f1=f1_score(yt, t),
                rec=recall_score(yt, t), prec=precision_score(yt, t, zero_division=0), bacc=balanced_accuracy_score(yt, t))


def sk(name, seed):
    return {'LR': LogisticRegression(max_iter=3000, class_weight='balanced'),
            'RF': RandomForestClassifier(300, class_weight='balanced', n_jobs=-1, random_state=seed),
            'MLP': MLPClassifier((32, 16), max_iter=300, early_stopping=True, random_state=seed)}[name]


def mlp_fit_predict(m, Xtr, ytr, Xte):
    # MLP has no class_weight: oversample positives to balance
    pos = np.where(ytr == 1)[0]; r = np.random.default_rng(0)
    extra = r.choice(pos, size=(ytr == 0).sum() - len(pos), replace=True)
    idx = np.concatenate([np.arange(len(ytr)), extra])
    m.fit(Xtr[idx], ytr[idx]); return m.predict_proba(Xte)[:, 1]


def run_classical(name, Xtr, ytr, Xte, seed):
    m = sk(name, seed)
    if name == 'MLP':
        return mlp_fit_predict(m, Xtr, ytr, Xte)
    m.fit(Xtr, ytr); return m.predict_proba(Xte)[:, 1]


def split_iter(protocol, A):
    if protocol == 'temporal':
        for i, trb, te in folds(X):
            yield i, trb, te
    else:
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        for i, (tr, te) in enumerate(skf.split(A, y)):
            yield i, np.array_split(tr, 4), te


rows = []
mode = sys.argv[1] if __name__ == "__main__" else None

if mode == 'E1':
    A = X[S['clean']].values
    for i, trb, te in folds(X):
        tr = np.concatenate(trb)
        P = Pipe().fit(A[tr]); Ztr, Zte = P.transform(A[tr]), P.transform(A[te])
        for name in ['LR', 'RF', 'MLP']:
            for s in SEEDS if name != 'LR' else [0]:
                rows.append(dict(exp='E1', model=name, feats='full', q=len(S['clean']), fold=i, seed=s,
                                 **metrics(y[te], run_classical(name, Ztr, y[tr], Zte, s))))
        for q in [4, 6, 8]:
            Pq = Pipe(q).fit(A[tr]); Qtr, Qte = Pq.transform(A[tr]), Pq.transform(A[te])
            Pa = Pipe(q, True).fit(A[tr]); Atr, Ate = Pa.transform(A[tr]), Pa.transform(A[te])
            for name in ['LR', 'RF', 'MLP']:
                for s in SEEDS if name != 'LR' else [0]:
                    rows.append(dict(exp='E1', model=name, feats='pca', q=q, fold=i, seed=s,
                                     **metrics(y[te], run_classical(name, Qtr, y[tr], Qte, s))))
            for s in SEEDS:
                m = VQC(q, L_, s); train_central(m, Atr, y[tr], EP, BS, LR_, s)
                rows.append(dict(exp='E1', model='VQC', feats='pca', q=q, fold=i, seed=s, nparams=m.params.size,
                                 **metrics(y[te], m.predict_proba(Ate))))
            print('fold', i, 'q', q, 'done', flush=True)

if mode == 'E2':
    for fs in ['all', 'clean']:
        A = X[S[fs]].values
        for protocol in ['random', 'temporal']:
            for i, trb, te in split_iter(protocol, A):
                tr = np.concatenate(trb)
                Pa = Pipe(4, True).fit(A[tr]); Atr, Ate = Pa.transform(A[tr]), Pa.transform(A[te])
                for s in SEEDS:
                    m = VQC(4, L_, s); train_central(m, Atr, y[tr], EP, BS, LR_, s)
                    rows.append(dict(exp='E2', model='VQC', feats=fs, protocol=protocol, fold=i, seed=s,
                                     **metrics(y[te], m.predict_proba(Ate))))
                P = Pipe().fit(A[tr]); Ztr, Zte = P.transform(A[tr]), P.transform(A[te])
                rows.append(dict(exp='E2', model='RF', feats=fs, protocol=protocol, fold=i, seed=0,
                                 **metrics(y[te], run_classical('RF', Ztr, y[tr], Zte, 0))))
            print(fs, protocol, 'done', flush=True)

if mode == 'E3':
    A = X[S['clean']].values
    for i, trb, te in folds(X):
        tr = np.concatenate(trb)
        Pa = Pipe(4, True).fit(A[tr]); Pq = Pipe(4).fit(A[tr]); Pf = Pipe().fit(A[tr])
        feats = {'VQC': (Pa, lambda s: VQC(4, L_, s)),
                 'LinPCA': (Pq, lambda s: LinearModel(4, s)),
                 'LinFull': (Pf, lambda s: LinearModel(len(S['clean']), s))}
        for name, (P, mk) in feats.items():
            clients = [(P.transform(A[b]), y[b]) for b in trb]
            Xte = P.transform(A[te])
            for sigma in [0.0, 0.6, 1.0, 1.5]:
                for s in SEEDS:
                    m = mk(s)
                    lr = LR_ if name == 'VQC' else 0.02
                    info = train_fedavg(m, clients, rounds=EP, local_epochs=1, bs=BS, lr=lr, seed=s,
                                        clip=(1.0 if sigma > 0 else None), sigma=sigma)
                    eps = max(epsilon(c['q'], sigma, c['steps']) for c in info) if sigma > 0 else float('inf')
                    rows.append(dict(exp='E3', model=name, sigma=sigma, eps=eps, fold=i, seed=s,
                                     **metrics(y[te], m.predict_proba(Xte))))
            # centralised reference with identical model/optimiser (no DP)
            for s in SEEDS:
                m = mk(s); lr = LR_ if name == 'VQC' else 0.02
                train_central(m, P.transform(A[tr]), y[tr], EP, BS, lr, s)
                rows.append(dict(exp='E3', model=name, sigma=-1, eps=float('inf'), fold=i, seed=s,
                                 **metrics(y[te], m.predict_proba(Xte))))
        print('fold', i, 'done', flush=True)

pd.DataFrame(rows).to_csv(f'res_{mode}.csv', index=False)
print('saved', len(rows))

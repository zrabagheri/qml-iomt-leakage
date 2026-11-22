"""Centralised, federated (FedAvg) and DP-SGD training shared by the VQC and the linear model."""
import numpy as np
from math import comb, log
from scipy.special import logsumexp


class Adam:
    def __init__(self, n, lr):
        self.m = np.zeros(n); self.v = np.zeros(n); self.t = 0; self.lr = lr

    def step(self, p, g, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        self.m = b1 * self.m + (1 - b1) * g
        self.v = b2 * self.v + (1 - b2) * g * g
        mh = self.m / (1 - b1 ** self.t); vh = self.v / (1 - b2 ** self.t)
        return p - self.lr * mh / (np.sqrt(vh) + eps)


def local_train(model, X, y, wpos, steps, bs, lr, rng, clip=None, sigma=0.0, p0=None):
    """Mini-batch training; if clip is set, DP-SGD (per-example clipping + Gaussian noise)."""
    p = model.params.copy() if p0 is None else p0.copy()
    opt = Adam(p.size, lr)
    n = len(y)
    for _ in range(steps):
        idx = rng.choice(n, size=min(bs, n), replace=False)
        g, _ = model.per_example_grads(X[idx], y[idx], wpos, p)
        if clip is not None:
            nrm = np.linalg.norm(g, axis=1, keepdims=True)
            g = g / np.maximum(1.0, nrm / clip)
            gs = g.sum(0) + rng.normal(0, sigma * clip, size=p.size)
            gbar = gs / len(idx)
        else:
            gbar = g.mean(0)
        p = opt.step(p, gbar)
    return p


def train_central(model, X, y, epochs, bs, lr, seed, clip=None, sigma=0.0):
    rng = np.random.default_rng(seed)
    wpos = (y == 0).sum() / max(1, (y == 1).sum())
    steps = int(np.ceil(epochs * len(y) / bs))
    model.params = local_train(model, X, y, wpos, steps, bs, lr, rng, clip, sigma)
    return dict(steps=steps, q=bs / len(y))


def train_fedavg(model, clients, rounds, local_epochs, bs, lr, seed, clip=None, sigma=0.0):
    """clients: list of (X, y). FedAvg weighted by client size; all clients join every round."""
    rng = np.random.default_rng(seed)
    ytot = np.concatenate([c[1] for c in clients])
    wpos = (ytot == 0).sum() / max(1, (ytot == 1).sum())
    sizes = np.array([len(c[1]) for c in clients], float)
    info = []
    for _ in range(rounds):
        new = []
        for (Xc, yc) in clients:
            steps = max(1, int(np.ceil(local_epochs * len(yc) / bs)))
            new.append(local_train(model, Xc, yc, wpos, steps, bs, lr, rng, clip, sigma, p0=model.params))
        model.params = np.average(np.stack(new), axis=0, weights=sizes)
    for (Xc, yc) in clients:
        steps = max(1, int(np.ceil(local_epochs * len(yc) / bs)))
        info.append(dict(steps=steps * rounds, q=min(bs, len(yc)) / len(yc)))
    return info


# ---------- Renyi-DP accountant for the sampled Gaussian mechanism (integer orders) ----------
ORDERS = list(range(2, 257))


def _rdp_sgm(q, sigma, alpha):
    if q >= 1.0:
        return alpha / (2 * sigma ** 2)
    terms = [log(comb(alpha, i)) + i * log(q) + (alpha - i) * log(1 - q) + (i * i - i) / (2 * sigma ** 2)
             for i in range(alpha + 1)]
    return logsumexp(terms) / (alpha - 1)


def epsilon(q, sigma, steps, delta=1e-5):
    if sigma <= 0:
        return float('inf')
    best = float('inf')
    for a in ORDERS:
        r = steps * _rdp_sgm(q, sigma, a)
        best = min(best, r + log(1 / delta) / (a - 1))
    return best

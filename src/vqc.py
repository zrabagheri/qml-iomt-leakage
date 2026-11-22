"""Variational quantum classifier simulated exactly with a NumPy statevector.

Circuit (data re-uploading, real amplitudes):
  |0>^q -> for l in 1..L: [RY(x_j) on each qubit j] -> [RY(theta_lj) on each qubit] -> CNOT ring
  readout <Z_0>;  logit = w * <Z_0> + b
Per-sample gradients of <Z_0> w.r.t. theta are computed exactly with the adjoint method
(expval_and_grad_adjoint, used for training) and checked against the parameter-shift rule
(expval_and_grad); per-sample gradients are also what DP-SGD needs.
"""
import numpy as np


def _ry(state, q, ang):
    """Apply RY(ang) to qubit q. state: (B, 2,...,2); ang: scalar or (B,)."""
    c = np.cos(ang / 2.0); s = np.sin(ang / 2.0)
    st = np.moveaxis(state, q + 1, 1)
    a0, a1 = st[:, 0], st[:, 1]
    if np.ndim(ang):
        sh = (-1,) + (1,) * (a0.ndim - 1)
        c = c.reshape(sh); s = s.reshape(sh)
    n0 = c * a0 - s * a1
    n1 = s * a0 + c * a1
    out = np.stack([n0, n1], axis=1)
    return np.moveaxis(out, 1, q + 1)


def _cnot(state, ctl, tgt):
    st = state.copy()
    idx1 = [slice(None)] * state.ndim
    idx1[ctl + 1] = 1
    sub = st[tuple(idx1)]                      # control = 1
    t = tgt + 1 if tgt < ctl else tgt          # axis index after removing control axis
    sub = np.flip(sub, axis=t)
    st[tuple(idx1)] = sub
    return st


def forward(X, theta, q, L):
    """X: (B,q) encoding angles; theta: (L,q). Returns <Z_0> per sample, shape (B,)."""
    B = X.shape[0]
    state = np.zeros((B,) + (2,) * q); state[(slice(None),) + (0,) * q] = 1.0
    for l in range(L):
        for j in range(q):
            state = _ry(state, j, X[:, j])
        for j in range(q):
            state = _ry(state, j, theta[l, j])
        if q > 1:
            for j in range(q):
                state = _cnot(state, j, (j + 1) % q)
    p = state ** 2
    p0 = p.reshape(B, 2, -1)[:, 0].sum(1)
    return 2 * p0 - 1


def expval_and_grad(X, theta, q, L):
    """<Z_0> (B,) and per-sample d<Z_0>/dtheta (B, L*q) by the parameter-shift rule."""
    z = forward(X, theta, q, L)
    G = np.zeros((X.shape[0], theta.size))
    for k in range(theta.size):
        tp = theta.copy().ravel(); tm = tp.copy()
        tp[k] += np.pi / 2; tm[k] -= np.pi / 2
        G[:, k] = 0.5 * (forward(X, tp.reshape(theta.shape), q, L) - forward(X, tm.reshape(theta.shape), q, L))
    return z, G


class VQC:
    """params vector = [theta (L*q), w, b]."""

    def __init__(self, q, L, seed=0):
        self.q, self.L = q, L
        r = np.random.default_rng(seed)
        self.params = np.concatenate([r.uniform(-0.1, 0.1, L * q), [2.0, 0.0]])

    def theta(self, p=None):
        p = self.params if p is None else p
        return p[:-2].reshape(self.L, self.q)

    def logit(self, X, p=None):
        p = self.params if p is None else p
        return p[-2] * forward(X, self.theta(p), self.q, self.L) + p[-1]

    def predict_proba(self, X, bs=2048):
        return np.concatenate([1 / (1 + np.exp(-self.logit(X[i:i + bs]))) for i in range(0, len(X), bs)])

    def per_example_grads(self, X, y, wpos, p=None):
        """Weighted BCE per-example gradients (B, P) and loss."""
        p = self.params if p is None else p
        z, G = expval_and_grad_adjoint(X, self.theta(p), self.q, self.L)
        lg = p[-2] * z + p[-1]
        pr = 1 / (1 + np.exp(-lg))
        wt = np.where(y == 1, wpos, 1.0)
        dl = wt * (pr - y)                     # dLoss/dlogit
        g = np.concatenate([dl[:, None] * p[-2] * G, (dl * z)[:, None], dl[:, None]], axis=1)
        loss = -(wt * (y * np.log(pr + 1e-12) + (1 - y) * np.log(1 - pr + 1e-12))).mean()
        return g, loss


class LinearModel:
    """Logistic regression with the same training/DP/federated code path as the VQC."""

    def __init__(self, d, seed=0):
        self.params = np.zeros(d + 1)

    def logit(self, X, p=None):
        p = self.params if p is None else p
        return X @ p[:-1] + p[-1]

    def predict_proba(self, X):
        return 1 / (1 + np.exp(-self.logit(X)))

    def per_example_grads(self, X, y, wpos, p=None):
        p = self.params if p is None else p
        pr = 1 / (1 + np.exp(-self.logit(X, p)))
        wt = np.where(y == 1, wpos, 1.0)
        dl = wt * (pr - y)
        g = np.concatenate([dl[:, None] * X, dl[:, None]], axis=1)
        loss = -(wt * (y * np.log(pr + 1e-12) + (1 - y) * np.log(1 - pr + 1e-12))).mean()
        return g, loss


def expval_and_grad_adjoint(X, theta, q, L):
    """Same outputs as expval_and_grad, via the adjoint (reverse-mode) method for real orthogonal gates."""
    B = X.shape[0]
    gates = []                                 # (kind, qubit/ctl, angle or tgt, param index or None)
    for l in range(L):
        for j in range(q):
            gates.append(('ry', j, X[:, j], None))
        for j in range(q):
            gates.append(('ry', j, theta[l, j], l * q + j))
        if q > 1:
            for j in range(q):
                gates.append(('cx', j, (j + 1) % q, None))
    psi = np.zeros((B,) + (2,) * q); psi[(slice(None),) + (0,) * q] = 1.0
    for g in gates:
        psi = _ry(psi, g[1], g[2]) if g[0] == 'ry' else _cnot(psi, g[1], g[2])
    zsign = np.ones((2,) + (1,) * (q - 1)); zsign[1] = -1
    lam = psi * zsign                          # O|psi>
    z = (psi * lam).reshape(B, -1).sum(1)
    G = np.zeros((B, theta.size))
    for g in reversed(gates):
        if g[0] == 'ry':
            ang = g[2]
            psi = _ry(psi, g[1], -ang if not np.ndim(ang) else -ang)   # psi_{k-1}
            if g[3] is not None:
                d = _ry(psi, g[1], ang + np.pi)                      # 2 * dU/dtheta psi_{k-1}
                G[:, g[3]] = (lam * d).reshape(B, -1).sum(1)
            lam = _ry(lam, g[1], -ang)
        else:
            psi = _cnot(psi, g[1], g[2]); lam = _cnot(lam, g[1], g[2])
    return z, G

"""Adjoint gradients vs parameter-shift gradients (paper: max difference below 1e-14)."""
import numpy as np
from vqc import expval_and_grad, expval_and_grad_adjoint
r = np.random.default_rng(0); m = 0.0
for q, L in [(2, 2), (4, 4), (6, 4), (8, 4)]:
    X = r.uniform(-3, 3, (16, q)); th = r.uniform(-3, 3, (L, q))
    z1, G1 = expval_and_grad(X, th, q, L); z2, G2 = expval_and_grad_adjoint(X, th, q, L)
    m = max(m, np.abs(G1 - G2).max(), np.abs(z1 - z2).max())
print('max |adjoint - parameter shift| = %.2e' % m)

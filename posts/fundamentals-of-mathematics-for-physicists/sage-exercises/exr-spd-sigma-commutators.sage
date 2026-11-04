# Exercise exr-spd-sigma-commutators: the relations [Sigma^{ab}, gamma^c] and [Sigma^{ab}, Sigma^{cd}] in d = 5, and the
# dimension of the Lie algebra spanned by the Sigma^{ab}.
import numpy as np
from functools import reduce
from scipy.linalg import expm
s1 = np.array([[0, 1], [1, 0]], dtype=complex); s2 = np.array([[0, -1j], [1j, 0]]); s3 = np.array([[1, 0], [0, -1]], dtype=complex)
kron = lambda ms: reduce(np.kron, ms, np.eye(1))
def gammas(p, q):
    """p gamma matrices squaring to +1, then q squaring to -1 (tensor products of Pauli matrices)."""
    d = p + q; n = d // 2; G = []
    for k in range(n):
        for s in (s1, s2):
            G.append(kron([s3] * k + [s] + [np.eye(2)] * (n - k - 1)))
    if d % 2: G.append(kron([s3] * n))
    return G[:p] + [1j * x for x in G[p:]]
d = 5; G = gammas(d, 0); N = G[0].shape[0]; eta = np.eye(d)
Sig = lambda a, b: 0.25 * (G[a] @ G[b] - G[b] @ G[a])
comm = lambda x, y: x @ y - y @ x
r = range(d)
first = all(np.allclose(comm(Sig(a, b), G[c]), eta[b, c] * G[a] - eta[a, c] * G[b]) for a in r for b in r for c in r)
second = all(np.allclose(comm(Sig(a, b), Sig(c, e)), eta[b, c] * Sig(a, e) - eta[a, c] * Sig(b, e) - eta[b, e] * Sig(a, c) + eta[a, e] * Sig(b, c)) for a in r for b in r for c in r for e in r)
print("size N =", N, "; [Sigma, gamma] relation:", first, "; [Sigma, Sigma] relation:", second)
basis = [Sig(a, b) for a in r for b in r if a < b]
M = np.array([np.concatenate([x.real.ravel(), x.imag.ravel()]) for x in basis])
print("number of Sigma^{ab} with a < b:", len(basis), "; real rank:", np.linalg.matrix_rank(M))
print("anti-Hermitian and traceless:", all(np.allclose(x.conj().T, -x) and abs(np.trace(x)) < 1e-12 for x in basis))

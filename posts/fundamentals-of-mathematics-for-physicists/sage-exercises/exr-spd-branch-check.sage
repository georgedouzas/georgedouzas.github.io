# Exercise exr-spd-branch-check: Gamma^mu = gamma^mu x 1, Gamma^{3+a} = i gamma_5 x gamma_K^a are gamma matrices of
# signature (1, 3+n), and Gamma_* = gamma_5 x gamma_*^K, for n = 2, 4, 6.
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
weyl = [np.kron(s1, np.eye(2))] + [np.kron(1j * s2, s) for s in (s1, s2, s3)]
g5 = 1j * reduce(np.dot, weyl, np.eye(4))
print("gamma_5 = diag(-1,-1,1,1):", np.allclose(g5, np.diag([-1, -1, 1, 1])))
for n in (2, 4, 6):
    K = gammas(n, 0); NK = K[0].shape[0]
    Gam = [np.kron(x, np.eye(NK)) for x in weyl] + [np.kron(1j * g5, k) for k in K]
    D = 4 + n; q = D - 1; eta = [1] + [-1] * q; NN = 4 * NK
    ok = all(np.allclose(Gam[a] @ Gam[b] + Gam[b] @ Gam[a], 2 * (eta[a] if a == b else 0) * np.eye(NN)) for a in range(D) for b in range(D))
    Gst = (1j) ** ((D * (D - 1)) // 2 + q) * reduce(np.dot, Gam, np.eye(NN))
    gK = (1j) ** (n // 2) * reduce(np.dot, K, np.eye(NK))
    print("n = %d, D = %2d: size %d, Clifford relations: %s, Gamma_* = gamma_5 x gamma_*^K: %s" % (n, D, NN, ok, np.allclose(Gst, np.kron(g5, gK))))

# Exercise exr-spd-weights-chirality: weights h_k = -i Sigma^{2k-1,2k} of the Dirac spinor of Spin(2 nu) and the rule
# gamma_* = (-1)^nu prod_k (2 h_k).
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
for nu in (2, 3, 4, 5):
    G = gammas(2 * nu, 0); N = 2 ** nu
    H = [-1j * 0.5 * G[2 * k] @ G[2 * k + 1] for k in range(nu)]
    gst = (1j) ** nu * reduce(np.dot, G, np.eye(N))
    W = [tuple(float(H[k][i, i].real) for k in range(nu)) for i in range(N)]
    ch = [int(round(gst[i, i].real)) for i in range(N)]
    rule = all(c == (-1) ** nu * round(np.prod([2 * x for x in w])) for w, c in zip(W, ch))
    print("nu = %d: diagonal: %s; rule holds: %s; weights with gamma_* = +1: %d, = -1: %d" % (nu,
          all(np.allclose(h, np.diag(np.diag(h))) for h in H + [gst]), rule, ch.count(1), ch.count(-1)))
    if nu == 2: print("   Spin(4) weights and chiralities:", list(zip(W, ch)))

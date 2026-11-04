# Exercise exr-spd-four-pi: S(phi) = exp(phi/2 gamma^a gamma^b) in R^4 is -1 at phi = 2 pi and 1 at 4 pi, while the
# rotation Lambda(phi) it covers is already the identity at 2 pi.
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
G = gammas(4, 0); N = 4
def S(a, b, phi): return expm(phi / 2 * G[a] @ G[b])
def Lam(Sm):
    Si = np.linalg.inv(Sm)
    return np.array([[np.trace(Si @ G[c] @ Sm @ G[d]).real / N for d in range(4)] for c in range(4)])
for a in range(4):
    for b in range(a + 1, 4):
        print("plane (%d,%d): S(2pi) = -1: %s, S(4pi) = 1: %s, Lambda(2pi) = 1: %s" % (a + 1, b + 1,
              np.allclose(S(a, b, 2 * np.pi), -np.eye(N)), np.allclose(S(a, b, 4 * np.pi), np.eye(N)), np.allclose(Lam(S(a, b, 2 * np.pi)), np.eye(4))))
print("Lambda(pi/2) in the plane (1,2):"); print(np.round(Lam(S(0, 1, np.pi / 2)), 12))

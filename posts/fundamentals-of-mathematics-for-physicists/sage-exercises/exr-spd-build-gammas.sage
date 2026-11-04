# Exercise exr-spd-build-gammas: gamma matrices of every signature (p, q), p + q <= 10, by tensor products of Pauli
# matrices; check the Clifford relations and the hermiticity (gamma^a)^dagger = eta^{aa} gamma^a.
import numpy as np
from functools import reduce
s1 = np.array([[0, 1], [1, 0]], dtype=complex); s2 = np.array([[0, -1j], [1j, 0]]); s3 = np.array([[1, 0], [0, -1]], dtype=complex)
kron = lambda ms: reduce(np.kron, ms, np.eye(1))
def gammas(p, q):
    d = p + q; n = d // 2; G = []
    for k in range(n):
        for s in (s1, s2):
            G.append(kron([s3] * k + [s] + [np.eye(2)] * (n - k - 1)))
    if d % 2: G.append(kron([s3] * n))
    return G[:p] + [1j * x for x in G[p:]]
for d in range(1, 11):
    ok = True
    for p in range(d + 1):
        q = d - p; G = gammas(p, q); N = G[0].shape[0]; eta = [1] * p + [-1] * q
        ok &= all(np.allclose(G[a] @ G[b] + G[b] @ G[a], 2 * (eta[a] if a == b else 0) * np.eye(N)) for a in range(d) for b in range(d))
        ok &= all(np.allclose(G[a].conj().T, eta[a] * G[a]) for a in range(d))
    print("d = %2d: size N = %2d, relations and hermiticity hold for all %d signatures: %s" % (d, gammas(d, 0)[0].shape[0], d + 1, ok))

# Exercise exr-spd-spin-four: Spin(4) = SU(2) x SU(2). The combinations J^pm_a = -(i/2)(Sigma^{bc} pm Sigma^{a4})
# (a, b, c cyclic) close on two commuting su(2) algebras, each acting on one chirality only.
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
Sig = lambda a, b: 0.25 * (G[a] @ G[b] - G[b] @ G[a])
gst = (1j) ** 2 * reduce(np.dot, G, np.eye(N))
cyc = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
J = {sgn: [-0.5j * (Sig(b, c) + sgn * Sig(a, 3)) for (a, b, c) in cyc] for sgn in (1, -1)}
comm = lambda x, y: x @ y - y @ x
for sgn in (1, -1):
    ok = all(np.allclose(comm(J[sgn][a], J[sgn][b]), 1j * J[sgn][c]) for (a, b, c) in cyc)
    print("J^%s: [J_a, J_b] = i eps_abc J_c: %s" % ("+" if sgn > 0 else "-", ok))
print("[J^+_a, J^-_b] = 0 for all a, b:", all(np.allclose(comm(J[1][a], J[-1][b]), 0) for a in range(3) for b in range(3)))
for sgn in (1, -1):
    Pp, Pm = (np.eye(N) + gst) / 2, (np.eye(N) - gst) / 2
    print("J^%s vanishes on gamma_* = +1: %s, on gamma_* = -1: %s; Casimir on its nonzero block: %s" % ("+" if sgn > 0 else "-",
          all(np.allclose(x @ Pp, 0) for x in J[sgn]), all(np.allclose(x @ Pm, 0) for x in J[sgn]),
          np.round(np.linalg.eigvals(sum(x @ x for x in J[sgn])).real, 6)))

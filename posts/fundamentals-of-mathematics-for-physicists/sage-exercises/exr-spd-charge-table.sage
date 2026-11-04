# Exercise exr-spd-charge-table: solve (gamma^a)^T C + eta_C C gamma^a = 0 for the Euclidean gamma matrices in d = 1..8
# and read off epsilon_C from C^T = -epsilon_C C (the signs depend only on d mod 8).
import numpy as np
from functools import reduce
s1 = np.array([[0, 1], [1, 0]], dtype=complex); s2 = np.array([[0, -1j], [1j, 0]]); s3 = np.array([[1, 0], [0, -1]], dtype=complex)
kron = lambda ms: reduce(np.kron, ms, np.eye(1))
def euclid(d):
    n = d // 2; G = []
    for k in range(n):
        for s in (s1, s2):
            G.append(kron([s3] * k + [s] + [np.eye(2)] * (n - k - 1)))
    if d % 2: G.append(kron([s3] * n))
    return G
def solve(G, etaC):
    N = G[0].shape[0]; I = np.eye(N)
    # vec(A X B) = (B^T kron A) vec(X) with column-major vec
    M = np.vstack([np.kron(I, g.T) + etaC * np.kron(g.T, I) for g in G])
    u, sv, vh = np.linalg.svd(M)
    null = [vh[k].conj() for k in range(len(sv)) if sv[k] < 1e-9] + [vh[k].conj() for k in range(len(sv), N * N)]
    return [v.reshape((N, N), order="F") for v in null]
for d in range(1, 9):
    row = []
    for etaC in (1, -1):
        sols = solve(euclid(d), etaC)
        if not sols: row.append("eta_C = %+d: none" % etaC); continue
        C = sols[0]; eps = -1 if np.allclose(C.T, C) else (1 if np.allclose(C.T, -C) else None)
        row.append("eta_C = %+d: %d solution(s), epsilon_C = %+d" % (etaC, len(sols), eps))
    print("d = %d | " % d + " | ".join(row))

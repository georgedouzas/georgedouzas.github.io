# Exercise exr-spd-eleven-check: gamma matrices of signature (1,10); charge conjugation C with (gamma^a)^T = -C gamma^a C^-1
# is antisymmetric with C gamma^a symmetric; a reality matrix B with B^* B = +1 exists (Majorana spinors).
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
G = gammas(1, 10); N = G[0].shape[0]; eta = [1] + [-1] * 10
print("size", N, "; Clifford relations:", all(np.allclose(G[a] @ G[b] + G[b] @ G[a], 2 * (eta[a] if a == b else 0) * np.eye(N)) for a in range(11) for b in range(11)))
sym = [x for x in G if np.allclose(x.T, x)]; anti = [x for x in G if np.allclose(x.T, -x)]
for C in (reduce(np.dot, sym, np.eye(N)), reduce(np.dot, anti, np.eye(N))):
    if all(np.allclose(x.T @ C, -C @ x) for x in G):
        print("C = product of the", "symmetric" if C is not None and np.allclose(C, reduce(np.dot, sym, np.eye(N))) else "antisymmetric", "gammas;", end=" ")
        print("C found: antisymmetric", np.allclose(C.T, -C), "; every C gamma^a symmetric", all(np.allclose((C @ x).T, C @ x) for x in G))
real = [x for x in G if np.allclose(x.conj(), x)]; imag = [x for x in G if np.allclose(x.conj(), -x)]
for B in (reduce(np.dot, real, np.eye(N)), reduce(np.dot, imag, np.eye(N))):
    for s in (1, -1):
        if all(np.allclose(x.conj() @ B, s * B @ x) for x in G):
            print("B with s = %+d (product of %d gammas): B^* B = %+d * 1" % (s, len(real) if B is not None and np.allclose(B, reduce(np.dot, real, np.eye(N))) else len(imag), int(np.real((B.conj() @ B)[0, 0]))))

# Exercise exr-spd-bb-sign: the sign of B^* B for the matrices B with (gamma^a)^* = s B gamma^a B^{-1}, for every
# signature (p, q) with p + q <= 8; Majorana spinors exist iff some sign is +1, iff p - q = 0, 1, 2, 6, 7 mod 8.
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
allok = True
for d in range(1, 9):
    line = []
    for p in range(d + 1):
        q = d - p; G = gammas(p, q); N = G[0].shape[0]
        real = [x for x in G if np.allclose(x.conj(), x)]; imag = [x for x in G if np.allclose(x.conj(), -x)]
        signs = []
        for B in (reduce(np.dot, real, np.eye(N)), reduce(np.dot, imag, np.eye(N))):
            for s in (1, -1):
                if all(np.allclose(x.conj() @ B, s * B @ x) for x in G):
                    signs.append(int(np.sign(np.real((B.conj() @ B)[0, 0]))))
        majorana = 1 in signs
        allok &= majorana == ((p - q) % 8 in (0, 1, 2, 6, 7))
        line.append("(%d,%d):%s" % (p, q, "M" if majorana else "-"))
    print("d = %d  " % d + " ".join(line))
print("agrees with p - q mod 8 in {0, 1, 2, 6, 7}:", allok)

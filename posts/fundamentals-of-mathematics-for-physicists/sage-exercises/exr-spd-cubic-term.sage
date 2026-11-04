# Exercise exr-spd-cubic-term: the cubic term C = -(i/8) f_abc gamma^a gamma^b gamma^c of the left-invariant Dirac
# operator on SU(2) x SU(2), f = 2 eps on each factor, with Euclidean gamma matrices of dimension 6.
import itertools
import numpy as np
from functools import reduce
s1 = np.array([[0, 1], [1, 0]], dtype=complex); s2 = np.array([[0, -1j], [1j, 0]]); s3 = np.array([[1, 0], [0, -1]], dtype=complex)
kron = lambda ms: reduce(np.kron, ms, np.eye(1))
G = []
for k in range(3):
    for s in (s1, s2):
        G.append(kron([s3] * k + [s] + [np.eye(2)] * (2 - k)))
f = np.zeros((6, 6, 6))
for off in (0, 3):
    for (a, b, c) in itertools.permutations(range(3)):
        f[off + a, off + b, off + c] = 2 * Permutation([a + 1, b + 1, c + 1]).sign()
C = -1j / 8 * sum(f[a, b, c] * G[a] @ G[b] @ G[c] for a in range(6) for b in range(6) for c in range(6))
gst = (1j) ** 3 * reduce(np.dot, G, np.eye(8))
print("C Hermitian:", np.allclose(C, C.conj().T), "; C anticommutes with gamma_*:", np.allclose(C @ gst, -gst @ C))
print("eigenvalues of C:", np.round(np.sort(np.linalg.eigvals(C).real), 6))
A = -1j / 8 * sum(f[a, b, c] * G[a] @ G[b] @ G[c] for a in range(3) for b in range(3) for c in range(3))
Bm = C - A
print("the two factors' terms anticommute:", np.allclose(A @ Bm, -Bm @ A), "; each squares to", np.round((A @ A)[0, 0].real, 6))
s = [s1, s2, s3]
c3 = -1j / 8 * sum(f[a, b, c] * s[a] @ s[b] @ s[c] for a in range(3) for b in range(3) for c in range(3))
print("on a single SU(2) with gamma^a = sigma_a the cubic term is", np.round(c3[0, 0].real, 6), "times 1:", np.allclose(c3, c3[0, 0] * np.eye(2)))

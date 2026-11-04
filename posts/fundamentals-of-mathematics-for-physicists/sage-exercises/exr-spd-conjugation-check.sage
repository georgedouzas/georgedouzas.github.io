# Exercise exr-spd-conjugation-check: complex conjugation (via B) preserves or exchanges the chiralities according to
# (-1)^{(p - q)/2}, in Euclidean and Lorentzian signature, d = 2, ..., 10.
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
for (label, sig) in (("Euclidean", lambda d: (d, 0)), ("Lorentzian", lambda d: (1, d - 1))):
    for d in (2, 4, 6, 8, 10):
        p, q = sig(d); G = gammas(p, q); N = G[0].shape[0]
        real = [x for x in G if np.allclose(x.conj(), x)]; imag = [x for x in G if np.allclose(x.conj(), -x)]
        Bs = [B for B in (reduce(np.dot, real, np.eye(N)), reduce(np.dot, imag, np.eye(N))) if any(all(np.allclose(x.conj() @ B, s * B @ x) for x in G) for s in (1, -1))]
        B = Bs[0]; gst = (1j) ** ((d * (d - 1)) // 2 + q) * reduce(np.dot, G, np.eye(N))
        sign = 1 if np.allclose(B @ gst @ np.linalg.inv(B), gst.conj()) else (-1 if np.allclose(B @ gst @ np.linalg.inv(B), -gst.conj()) else 0)
        print("%-10s d = %2d (p - q = %3d): B gamma_* B^-1 = %+d gamma_*^*, (-1)^((p-q)/2) = %+d" % (label, d, p - q, sign, (-1) ** ((p - q) // 2)))

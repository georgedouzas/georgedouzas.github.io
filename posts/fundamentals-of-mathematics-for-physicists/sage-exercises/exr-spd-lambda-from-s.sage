# Exercise exr-spd-lambda-from-s: for a random omega in so(3), S = exp(omega_ab Sigma^{ab}/2) with gamma^a = sigma_a
# covers Lambda = exp(omega), recovered as Lambda^a_b = tr(S^{-1} sigma_a S sigma_b)/2.
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
np.random.seed(7)
A = np.random.randn(3, 3); omega = A - A.T
sig = [s1, s2, s3]
Sig = lambda a, b: 0.25 * (sig[a] @ sig[b] - sig[b] @ sig[a])
S = expm(0.5 * sum(omega[a, b] * Sig(a, b) for a in range(3) for b in range(3)))
Si = np.linalg.inv(S)
Lam = np.array([[0.5 * np.trace(Si @ sig[a] @ S @ sig[b]).real for b in range(3)] for a in range(3)])
print("omega ="); print(np.round(omega, 4))
print("Lambda from S ="); print(np.round(Lam, 6))
print("max |Lambda - expm(omega)| =", np.abs(Lam - expm(omega)).max())
print("Lambda orthogonal:", np.allclose(Lam @ Lam.T, np.eye(3)), "; det Lambda =", round(np.linalg.det(Lam), 12))

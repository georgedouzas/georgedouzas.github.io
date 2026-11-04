# Exercise exr-spd-cover-formula: Lambda_ab(S) = tr(S^{-1} sigma_a S sigma_b)/2 is a homomorphism SU(2) -> SO(3) with
# kernel {1, -1}; Lambda of exp(-i phi sigma_1/2) is the rotation by phi in the (2,3) plane.
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
sig = [s1, s2, s3]
Lam = lambda S: np.array([[0.5 * np.trace(np.linalg.inv(S) @ sig[a] @ S @ sig[b]).real for b in range(3)] for a in range(3)])
def random_su2(rng):
    v = rng.standard_normal(4); v /= np.linalg.norm(v)
    return np.array([[v[0] + 1j * v[3], v[2] + 1j * v[1]], [-v[2] + 1j * v[1], v[0] - 1j * v[3]]])
rng = np.random.default_rng(int(11))
S1, S2 = random_su2(rng), random_su2(rng)
print("S1, S2 in SU(2):", all(np.allclose(S.conj().T @ S, np.eye(2)) and np.isclose(np.linalg.det(S), 1) for S in (S1, S2)))
print("Lambda(S1 S2) = Lambda(S1) Lambda(S2):", np.allclose(Lam(S1 @ S2), Lam(S1) @ Lam(S2)))
print("Lambda(-S1) = Lambda(S1):", np.allclose(Lam(-S1), Lam(S1)), "; Lambda in SO(3):", np.allclose(Lam(S1) @ Lam(S1).T, np.eye(3)), round(np.linalg.det(Lam(S1)), 12))
phi = 0.7
print("Lambda(exp(-i phi sigma_1 / 2)) for phi = 0.7:"); print(np.round(Lam(expm(-0.5j * phi * s1)), 6))
print("cos(0.7), sin(0.7) =", round(np.cos(phi), 6), round(np.sin(phi), 6))

# Exercise exr-spd-six-su4: the 15 generators Sigma^{ab} of Spin(6) restricted to the chirality +1 subspace span su(4);
# on the chirality -1 subspace each generator has the opposite eigenvalues (the conjugate representation).
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
G = gammas(6, 0); N = 8
gst = (1j) ** 3 * reduce(np.dot, G, np.eye(N))
plus = [i for i in range(N) if gst[i, i].real > 0]; minus = [i for i in range(N) if gst[i, i].real < 0]
Sig = lambda a, b: 0.25 * (G[a] @ G[b] - G[b] @ G[a])
gens = [Sig(a, b) for a in range(6) for b in range(a + 1, 6)]
print("gamma_* diagonal:", np.allclose(gst, np.diag(np.diag(gst))), "; dims of S_+ and S_-:", len(plus), len(minus))
print("Sigma^{ab} preserve S_+:", all(np.allclose(x[np.ix_(minus, plus)], 0) for x in gens))
R = [x[np.ix_(plus, plus)] for x in gens]
print("restrictions anti-Hermitian and traceless:", all(np.allclose(r.conj().T, -r) and abs(np.trace(r)) < 1e-12 for r in R))
M = np.array([np.concatenate([r.real.ravel(), r.imag.ravel()]) for r in R])
print("real dimension of their span:", np.linalg.matrix_rank(M), "(dim su(4) = 15)")
Rm = [x[np.ix_(minus, minus)] for x in gens]
print("spectrum on S_- is minus the spectrum on S_+ for every generator:", all(np.allclose(np.sort_complex(np.linalg.eigvals(a)), np.sort_complex(-np.linalg.eigvals(b))) for a, b in zip(R, Rm)))

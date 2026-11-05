# Figure: the size N = 2^[d/2] of irreducible gamma matrices built by tensor products of Pauli matrices, for
# d = 1, ..., 12, together with the real dimension of a minimal spinor in Lorentzian signature (one plus sign).
# Output: figures/sd-spinor-dimensions.svg.
import os
import numpy as np
from functools import reduce
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sd-spinor-dimensions.svg")
s1 = np.array([[0, 1], [1, 0]], dtype=complex); s2 = np.array([[0, -1j], [1j, 0]]); s3 = np.array([[1, 0], [0, -1]], dtype=complex)
kron = lambda ms: reduce(np.kron, ms, np.eye(1))
def gammas(p, q):
    """p matrices squaring to +1 followed by q squaring to -1 (tensor products of Pauli matrices)."""
    d = p + q; n = d // 2; G = []
    for k in range(n):
        for s in (s1, s2):
            G.append(kron([s3] * k + [s] + [np.eye(2)] * (n - k - 1)))
    if d % 2: G.append(kron([s3] * n))
    return G[:p] + [1j * x for x in G[p:]]
def clifford_ok(p, q):
    G = gammas(p, q); N = G[0].shape[0]; eta = [1] * p + [-1] * q
    return all(np.allclose(G[a] @ G[b] + G[b] @ G[a], 2 * (eta[a] if a == b else 0) * np.eye(N)) for a in range(p + q) for b in range(p + q))
def minimal_real(d, pq):
    N = 2 ** (d // 2); r = pq % 8
    return N // 2 if r == 0 else (2 * N if r in (3, 5) else N)
ds = list(range(1, 13))
N = [2 ** (d // 2) for d in ds]
R = [minimal_real(d, 1 - (d - 1)) for d in ds]
P = list_plot(list(zip(ds, N)), plotjoined=True, marker="o", color="royalblue", legend_label=r"complex size $N = 2^{\lfloor d/2\rfloor}$ of $\gamma^a$", scale="semilogy", base=2)
P += list_plot(list(zip(ds, R)), plotjoined=True, marker="s", color="firebrick", linestyle="--", legend_label=r"real dimension of a minimal Lorentzian spinor", scale="semilogy", base=2)
P.axes_labels([r"$d$", ""]); P.set_legend_options(loc="upper left", font_size=8)
P.save(OUT, figsize=[6.5, 4], gridlines="major")
print("[PASS] the tensor-product matrices satisfy the Clifford relations for every (p, q) with p + q <= 12:", all(clifford_ok(p, d - p) for d in ds for p in range(d + 1)))
print("[PASS] their size is 2^[d/2]:", all(gammas(p, d - p)[0].shape[0] == 2 ** (d // 2) for d in ds for p in range(d + 1)))
print("[PASS] minimal real Lorentzian dimensions for d = 2, 3, 4, 6, 10, 11 are 1, 2, 4, 8, 16, 32:", [R[d - 1] for d in (2, 3, 4, 6, 10, 11)] == [1, 2, 4, 8, 16, 32])
print("wrote", os.path.relpath(OUT))

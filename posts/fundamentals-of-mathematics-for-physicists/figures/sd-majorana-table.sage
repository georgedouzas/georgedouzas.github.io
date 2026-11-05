# Figure: which reality conditions the spinors of signature (p, q) admit, for d = p + q = 1, ..., 11, computed from
# explicit matrices B with (gamma^a)^* = +-B gamma^a B^{-1}: Majorana-Weyl (MW), Majorana (M), symplectic
# Majorana-Weyl (SMW) or symplectic Majorana (SM); the number in each cell is the real dimension of a minimal spinor.
# Output: figures/sd-majorana-table.svg.
import os
import numpy as np
from functools import reduce
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sd-majorana-table.svg")
s1 = np.array([[0, 1], [1, 0]], dtype=complex); s2 = np.array([[0, -1j], [1j, 0]]); s3 = np.array([[1, 0], [0, -1]], dtype=complex)
kron = lambda ms: reduce(np.kron, ms, np.eye(1))
def gammas(p, q):
    d = p + q; n = d // 2; G = []
    for k in range(n):
        for s in (s1, s2):
            G.append(kron([s3] * k + [s] + [np.eye(2)] * (n - k - 1)))
    if d % 2: G.append(kron([s3] * n))
    return G[:p] + [1j * x for x in G[p:]]
def product(ms, N): return reduce(np.dot, ms, np.eye(N))
def kind(p, q):
    """Classify by the matrices B: every gamma here is real or imaginary, so B is the product of the real or of the
    imaginary ones; Majorana iff B^* B = +1 for some B, Majorana-Weyl iff such a B also maps gamma_* to gamma_*^*."""
    G = gammas(p, q); N = G[0].shape[0]; d = p + q
    real = [x for x in G if np.allclose(x.conj(), x)]; imag = [x for x in G if np.allclose(x.conj(), -x)]
    Bs = []
    for B in (product(real, N), product(imag, N)):
        for s in (1, -1):
            if all(np.allclose(x.conj() @ B, s * B @ x) for x in G):
                Bs.append((np.real((B.conj() @ B)[0, 0]), B))
    maj = [B for c, B in Bs if c > 0]
    gst = (1j) ** ((d * (d - 1)) // 2 + q) * product(G, N) if d % 2 == 0 else None
    weyl_real = lambda B: gst is not None and np.allclose(gst.conj() @ B, B @ gst)
    if maj:
        return "MW" if any(weyl_real(B) for B in maj) else "M"
    return "SMW" if d % 2 == 0 and any(weyl_real(B) for c, B in Bs) else "SM"
def rule(p, q):
    r = (p - q) % 8
    return "MW" if r == 0 else ("M" if r in (1, 2, 6, 7) else ("SMW" if r == 4 else "SM"))
def minimal_real(p, q):
    N = 2 ** ((p + q) // 2); k = rule(p, q)
    return N // 2 if k == "MW" else (2 * N if k == "SM" else N)
colours = {"MW": "#1e8449", "M": "#5dade2", "SMW": "#f5b041", "SM": "#d5d8dc"}
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
fig, ax = plt.subplots(figsize=(7, 6))
table = {}
for d in range(1, 12):
    for q in range(d + 1):
        k = kind(d - q, q); table[(d, q)] = k
        ax.add_patch(Rectangle((d - 0.5, q - 0.5), 1, 1, facecolor=colours[k], edgecolor="white"))
        ax.text(d, q, str(minimal_real(d - q, q)), ha="center", va="center", fontsize=7)
ax.set_xlim(0.5, 11.5); ax.set_ylim(-0.5, 11.5); ax.set_aspect("equal")
ax.set_xticks(range(1, 12)); ax.set_yticks(range(0, 12))
ax.set_xlabel(r"$d = p + q$"); ax.set_ylabel(r"$q$ (number of minus signs)")
ax.legend(handles=[Patch(facecolor=c, label=k) for k, c in colours.items()], loc="upper left", fontsize=8)
fig.tight_layout(); fig.savefig(OUT)
print("[PASS] the computed type agrees with the rule p - q mod 8 in {0: MW; 1, 2, 6, 7: M; 4: SMW; 3, 5: SM}:", all(table[(d, q)] == rule(d - q, q) for d in range(1, 12) for q in range(d + 1)))
print("[PASS] Lorentzian (one plus sign): MW in d = 2 and 10, M in d = 3, 4, 8, 9, 11:", [d for d in range(2, 12) if table[(d, d - 1)] == "MW"] == [2, 10] and [d for d in range(2, 12) if table[(d, d - 1)] == "M"] == [3, 4, 8, 9, 11])
print("[PASS] Euclidean (no minus sign): MW only in d = 8:", [d for d in range(1, 12) if table[(d, 0)] == "MW"] == [8])
print("wrote", os.path.relpath(OUT))

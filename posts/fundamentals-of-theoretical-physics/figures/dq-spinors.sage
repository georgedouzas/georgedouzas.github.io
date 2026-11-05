# Figure: the spin-up spinor u(p) for momentum p along z (m = 1, bar(u) u = 2m) in the Dirac representation, where its
# upper ("large") component is sqrt(E + m) and its lower ("small") one p/sqrt(E + m), and in the Weyl representation,
# where the left- and right-handed components are sqrt(E - p) and sqrt(E + p). Output: figures/dq-spinors.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dq-spinors.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
P = np.linspace(0, 4, 300); E = np.sqrt(P**2 + 1)
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
axs[0].plot(P, np.sqrt(E + 1), color="royalblue", lw=2, label=r"upper $\sqrt{E + m}$")
axs[0].plot(P, P / np.sqrt(E + 1), color="firebrick", lw=2, label=r"lower $p/\sqrt{E + m}$")
axs[0].set_title("Dirac representation", fontsize=10)
axs[1].plot(P, np.sqrt(E - P), color="royalblue", lw=2, label=r"left $\sqrt{E - p}$")
axs[1].plot(P, np.sqrt(E + P), color="firebrick", lw=2, label=r"right $\sqrt{E + p}$")
axs[1].set_title("Weyl representation", fontsize=10)
for a in axs:
    a.set_xlabel(r"$p/m$"); a.legend(fontsize=8, loc="upper left")
axs[0].set_ylabel(r"component (units of $\sqrt{m}$)")
fig.tight_layout(); fig.savefig(OUT)
s0 = identity_matrix(SR, 2); s1 = matrix(SR, [[0,1],[1,0]]); s2 = matrix(SR, [[0,-I],[I,0]]); s3 = matrix(SR, [[1,0],[0,-1]]); Z = matrix(SR, 2, 2)
blk = lambda a, b, c, d: block_matrix([[a, b], [c, d]], subdivide=False)
g = [blk(Z, s0, s0, Z)] + [blk(Z, s, -s, Z) for s in (s1, s2, s3)]
w = var("w", domain="real"); m = var("m", domain="positive"); E = m * (exp(w) + exp(-w)) / 2; p = m * (exp(w) - exp(-w)) / 2   # E = m cosh w, p = m sinh w
uW = sqrt(m) * vector(SR, [exp(-w / 2), 0, exp(w / 2), 0])   # sqrt(E -+ p) = sqrt(m) e^{-+w/2}
pslash = E * g[0] - p * g[3]   # gamma^mu p_mu with p_mu = (E, 0, 0, -p)
zz = lambda e: bool(SR(e).expand().simplify_full() == 0)
print("[PASS] (pslash - m) u = 0 for the Weyl-representation spinor:", all(zz(c) for c in (pslash - m) * uW))
U = matrix(SR, [[1, 0, 1, 0], [0, 1, 0, 1], [-1, 0, 1, 0], [0, -1, 0, 1]]) / sqrt(2)
uD = U * uW
ok = zz(uD[0]^2 - (E + m)) and zz(uD[2]^2 - p^2 / (E + m)) and bool(uD[1] == 0) and bool(uD[3] == 0)
print("[PASS] in the Dirac representation (U from the Weyl one) u = (sqrt(E + m), 0, p/sqrt(E + m), 0):", ok)
print("[PASS] bar(u) u = 2m and u^dag u = 2E:", zz(uW * g[0] * uW - 2 * m) and zz(uW * uW - 2 * E))
p = var("p", domain="positive"); E = sqrt(p^2 + m^2)
print("[PASS] the ratio of small to large component is p/(E + m), which is v/2 + O(v^3) for v = p/E:", bool(taylor((p / (E + m)).subs(p=m * var("v") / sqrt(1 - var("v")^2)), var("v"), 0, 2) == var("v") / 2))
print("wrote", os.path.relpath(OUT))

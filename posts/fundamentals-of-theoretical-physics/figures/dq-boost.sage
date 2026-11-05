# Figure: the spin-up rest spinor u(0) = sqrt(m)(1, 0, 1, 0) boosted along z with S = exp(w S^{03}): its left- and
# right-handed components scale as e^{-w/2} and e^{w/2}, the density psi^dag psi = 2m cosh w grows like the time
# component of a vector, and the scalar bar(psi) psi = 2m stays fixed (m = 1). Output: figures/dq-boost.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dq-boost.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
W = np.linspace(-2.5, 2.5, 300)
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(W, np.exp(-W / 2), color="royalblue", lw=2, label=r"left component $e^{-w/2}$")
ax.plot(W, np.exp(W / 2), color="firebrick", lw=2, label=r"right component $e^{w/2}$")
ax.plot(W, 2 * np.cosh(W), color="darkgreen", lw=1.5, ls="--", label=r"$\psi^\dagger\psi = 2m\cosh w$")
ax.plot(W, 2 + 0 * W, color="black", lw=1.5, ls=":", label=r"$\bar\psi\psi = 2m$")
ax.set_ylim(0, 6); ax.set_xlabel(r"rapidity $w$"); ax.set_ylabel(r"value (units of $\sqrt{m}$ or $m$)")
ax.legend(loc="upper center", fontsize=8); fig.tight_layout(); fig.savefig(OUT)
s0 = identity_matrix(SR, 2); s1 = matrix(SR, [[0,1],[1,0]]); s2 = matrix(SR, [[0,-I],[I,0]]); s3 = matrix(SR, [[1,0],[0,-1]]); Z = matrix(SR, 2, 2)
blk = lambda a, b, c, d: block_matrix([[a, b], [c, d]], subdivide=False)
g = [blk(Z, s0, s0, Z)] + [blk(Z, s, -s, Z) for s in (s1, s2, s3)]
w = var("w", domain="real")
S = (w * (g[0] * g[3] - g[3] * g[0]) / 4).exp()
u0 = vector(SR, [1, 0, 1, 0])
u = (S * u0).simplify_full()
print("[PASS] S u(0) = (e^{-w/2}, 0, e^{w/2}, 0):", all(bool(a == b) for a, b in zip(u, [exp(-w / 2), 0, exp(w / 2), 0])))
u = vector(SR, [exp(-w / 2), 0, exp(w / 2), 0]); dens = (u * u).simplify_full(); scal = (u * g[0] * u).simplify_full()  # u is real for real w
print("[PASS] psi^dag psi = 2 cosh w and bar(psi) psi = 2:", bool((dens - exp(w) - exp(-w)).simplify_full() == 0) and bool(scal.simplify_full() == 2))
print("[PASS] S^-1 gamma^0 S = cosh w gamma^0 + sinh w gamma^3 (checked at w = 7/10):", (S.inverse() * g[0] * S - (cosh(w) * g[0] + sinh(w) * g[3])).subs(w=0.7).n().norm() < 1e-12)
print("[PASS] gamma^0 S^dag gamma^0 = S^-1:", (g[0] * S.conjugate_transpose() * g[0] - S.inverse()).subs(w=0.7).n().norm() < 1e-12)
print("wrote", os.path.relpath(OUT))

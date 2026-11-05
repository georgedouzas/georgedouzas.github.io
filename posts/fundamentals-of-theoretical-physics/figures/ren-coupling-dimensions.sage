# Figure: mass dimensions of couplings against the space-time dimension d. Output: figures/ren-coupling-dimensions.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ren-coupling-dimensions.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
var('d')
phi = (d - 2) / 2; psi = (d - 1) / 2; A = (d - 2) / 2
dims = {r"quartic $\lambda$": d - 4 * phi, r"cubic $g$": d - 3 * phi, r"gauge $g$ ($D_\mu = \partial_\mu + gA_\mu$)": 1 - A,
        r"Yukawa $y$": d - phi - 2 * psi, r"four-fermion $G_F$": d - 4 * psi, r"Newton $G$": 2 - d}
fig, ax = plt.subplots(figsize=(6.5, 4.2))
ds = np.linspace(2, 11, 50)
styles = ["-", "-", "-", "--", "-.", ":"]
for (lab, f), st in zip(dims.items(), styles):
    ax.plot(ds, [float(f.subs(d=x)) for x in ds], st, lw=1.8, label=lab + ": $%s$" % latex(f.expand()))
ax.axhline(0, color="black", lw=0.8)
for x in [3, 4, 6]: ax.axvline(x, color="gray", ls=":", lw=0.7)
ax.set_xlabel("space-time dimension $d$"); ax.set_ylabel("mass dimension of the coupling"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT)
print("[PASS] [lambda] = 4 - d, [g_cubic] = 3 - d/2, [g_gauge] = [y] = 2 - d/2, [G_F] = [G] = 2 - d:", bool(dims[r"quartic $\lambda$"] == 4 - d) and bool((dims[r"cubic $g$"] - (3 - d / 2)).expand() == 0) and bool((dims[r"gauge $g$ ($D_\mu = \partial_\mu + gA_\mu$)"] - (2 - d / 2)).expand() == 0) and bool((dims[r"Yukawa $y$"] - (2 - d / 2)).expand() == 0) and bool((dims[r"four-fermion $G_F$"] - (2 - d)).expand() == 0))
print("[PASS] the gauge and cubic couplings are dimensionless exactly at d = 4 and d = 6:", bool(solve(2 - d / 2 == 0, d)[0].rhs() == 4) and bool(solve(3 - d / 2 == 0, d)[0].rhs() == 6))
print("wrote", os.path.relpath(OUT))

# Figure: the seesaw. Two Weyl fermions with the symmetric Majorana mass matrix [[0, mD], [mD, M]] give two Majorana
# fermions with masses |(M -+ sqrt(M^2 + 4 mD^2))/2|; for M >> mD the light one is mD^2/M and the heavy one M.
# Masses in units of mD, log-log. Output: figures/dq-seesaw.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dq-seesaw.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
Ms = np.logspace(-2, 4, 400)
light = np.abs(Ms - np.sqrt(Ms**2 + 4)) / 2; heavy = (Ms + np.sqrt(Ms**2 + 4)) / 2
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.loglog(Ms, heavy, color="firebrick", lw=2, label="heavy mass")
ax.loglog(Ms, light, color="royalblue", lw=2, label="light mass")
ax.loglog(Ms, Ms, color="firebrick", ls=":", lw=1, label=r"$M$")
ax.loglog(Ms, 1 / Ms, color="royalblue", ls=":", lw=1, label=r"$m_D^2/M$")
ax.set_xlabel(r"$M/m_D$"); ax.set_ylabel(r"mass$/m_D$"); ax.set_ylim(1e-4, 1e4); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT)
mD, M, x = var("mD M x", domain="positive")
ev = matrix(SR, [[0, mD], [mD, M]]).eigenvalues()
print("[PASS] the eigenvalues of [[0, mD], [mD, M]] are (M -+ sqrt(M^2 + 4 mD^2))/2:", set(e.simplify_full() for e in ev) == set([(M - sqrt(M^2 + 4 * mD^2)) / 2, (M + sqrt(M^2 + 4 * mD^2)) / 2]))
lt = ((sqrt(M^2 + 4 * mD^2) - M) / 2).subs(M=mD / x)
print("[PASS] the light mass is mD^2/M - mD^4/M^3 + O(M^-5):", bool(taylor(lt, x, 0, 3) == mD * x - mD * x^3))
print("[PASS] product of the eigenvalues = -mD^2 (so light x heavy = mD^2) and sum of the eigenvalues = M:", bool(prod(ev).simplify_full() == -mD^2) and bool(sum(ev).simplify_full() == M))
print("wrote", os.path.relpath(OUT))

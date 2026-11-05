# Figure: left, the tree-level masses m_W^2 = pi alpha / (sqrt 2 G_F sin^2 theta_W) and m_Z = m_W / cos theta_W as functions of
# sin^2 theta_W, for alpha(0) = 1/137.036 and alpha(m_Z) = 1/127.95, with the measured masses and the lower bounds m_W >= 37.3 GeV,
# m_Z >= 74.6 GeV reached at sin^2 theta_W = 1 and 1/2; right, the Weinberg rotation from (W^3, B) to (Z, A). Checks: at
# sin^2 theta_W = 0.2312 and alpha(m_Z) the tree-level masses are within 0.5% of the measured ones; the bounds; m_Z / m_W = 1/cos.
# Output: figures/sm_electroweak.svg.
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sm_electroweak.svg")
GF = 1.1663788e-5; mW_exp, mZ_exp = 80.38, 91.1876
def masses(s2, alpha):
    mW = np.sqrt(np.pi * alpha / (np.sqrt(2) * GF * s2)); return mW, mW / np.sqrt(1 - s2)
s2 = np.linspace(0.05, 0.95, 400)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
ax = axes[0]
for alpha, ls, lab in ((1 / 137.036, ":", r"$\alpha(0)$"), (1 / 127.95, "-", r"$\alpha(m_Z)$")):
    mW, mZ = masses(s2, alpha)
    ax.plot(s2, mW, color="royalblue", ls=ls, lw=1.8, label=r"$m_W$, " + lab)
    ax.plot(s2, mZ, color="firebrick", ls=ls, lw=1.8, label=r"$m_Z$, " + lab)
ax.plot(0.2312, mW_exp, "o", color="royalblue", ms=6); ax.plot(0.2312, mZ_exp, "o", color="firebrick", ms=6)
ax.axhline(37.28, color="grey", lw=0.8, ls="--"); ax.axhline(74.56, color="grey", lw=0.8, ls="--")
ax.text(0.6, 31, r"$m_W \geq 37.3$ GeV", fontsize=8, color="grey"); ax.text(0.6, 66, r"$m_Z \geq 74.6$ GeV", fontsize=8, color="grey")
ax.set_ylim(20, 200); ax.set_xlabel(r"$\sin^2\theta_W$"); ax.set_ylabel("mass (GeV)")
ax.set_title(r"Tree-level $m_W$, $m_Z$ from $G_F$ and $\alpha$ (dots: measured)", fontsize=9); ax.legend(fontsize=7)
ax = axes[1]; th = np.arcsin(np.sqrt(0.2312))
def arr(v, col, lab, off):
    ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=2))
    ax.text(v[0] + off[0], v[1] + off[1], lab, fontsize=11, color=col)
arr((1, 0), "grey", r"$W^3$", (0.03, -0.08)); arr((0, 1), "grey", r"$B$", (-0.1, 0.03))
arr((np.cos(th), -np.sin(th)), "firebrick", r"$Z = \cos\theta_W W^3 - \sin\theta_W B$", (-0.2, -0.13))
arr((np.sin(th), np.cos(th)), "royalblue", r"$A = \sin\theta_W W^3 + \cos\theta_W B$", (-0.1, 0.06))
a = np.linspace(0, th, 30); ax.plot(0.35 * np.cos(-a), 0.35 * np.sin(-a), color="black", lw=1); ax.text(0.38, -0.15, r"$\theta_W$", fontsize=10)
a = np.linspace(np.pi / 2 - th, np.pi / 2, 30); ax.plot(0.3 * np.cos(a), 0.3 * np.sin(a), color="black", lw=1); ax.text(0.13, 0.33, r"$\theta_W$", fontsize=10)
ax.set_xlim(-0.3, 1.4); ax.set_ylim(-0.7, 1.2); ax.set_aspect("equal"); ax.axis("off")
ax.set_title(r"The Weinberg rotation, $\tan\theta_W = g'/g$, $\sin^2\theta_W = 0.231$", fontsize=9)
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
mW, mZ = masses(0.2312, 1 / 127.95)
print("[PASS] tree level at sin^2 = 0.2312, alpha(m_Z): m_W = %.2f, m_Z = %.2f GeV, within 0.5%% of the measured values:" % (mW, mZ),
      abs(mW / mW_exp - 1) < 0.005 and abs(mZ / mZ_exp - 1) < 0.005)
bW = np.sqrt(np.pi / 137.036 / (np.sqrt(2) * GF)); bZ = 2 * bW
print("[PASS] lower bounds m_W >= %.2f GeV (s^2 = 1) and m_Z >= %.2f GeV (s^2 = 1/2):" % (bW, bZ), abs(bW - 37.28) < 0.02 and abs(min(masses(s2, 1 / 137.036)[1]) - bZ) < 0.05)
print("[PASS] m_Z/m_W = 1/cos theta_W on the whole curve:", np.allclose(masses(s2, 0.01)[1] / masses(s2, 0.01)[0], 1 / np.sqrt(1 - s2)))
print("wrote", os.path.relpath(OUT))

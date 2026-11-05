# Figure: active and passive boosts. Left: a worldline at rest moved by the active boost L(0.5). Right: the same
# worldline with the axes of an observer moving with velocity -tanh(0.5), in which it moves with +tanh(0.5).
# Output: figures/sr-passive-active.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sr-passive-active.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
eta = 0.5; v = np.tanh(eta); Lb = 1.6
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
for a in ax:
    a.axhline(0, color="black", lw=1); a.axvline(0, color="black", lw=1)
    a.plot([-Lb, Lb], [-Lb, Lb], ":", color="darkorange"); a.plot([-Lb, Lb], [Lb, -Lb], ":", color="darkorange")
    a.set_xlim(-Lb, Lb); a.set_ylim(-Lb, Lb); a.set_aspect("equal"); a.set_xlabel("$x$"); a.set_ylabel("$t$")
s = np.linspace(-Lb, Lb, 2)
ax[0].plot([0.4, 0.4], [-Lb, Lb], color="black", lw=2, label="particle at rest")
ax[0].plot(0.4*np.cosh(eta) + v*s, s + 0.4*np.sinh(eta), color="royalblue", lw=2, label=r"after the active boost $L(\eta)$")
ax[0].legend(fontsize=8, loc="lower right"); ax[0].set_title("active", fontsize=10)
ax[1].plot([0.4, 0.4], [-Lb, Lb], color="black", lw=2, label="particle at rest")
ax[1].plot(-v*s, s, color="royalblue", lw=1.8, label=r"axes of an observer with velocity $-\tanh\eta$"); ax[1].plot(s, -v*s, color="royalblue", lw=1.8)
ax[1].legend(fontsize=8, loc="lower right"); ax[1].set_title("passive", fontsize=10)
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
Lp = np.array([[np.cosh(eta), np.sinh(eta)], [np.sinh(eta), np.cosh(eta)]])   # passive matrix for velocity -v is L(+eta)
u = Lp @ np.array([1., 0])
print("[PASS] in the passive frame the particle has velocity +tanh(eta):", abs(u[1]/u[0] - v) < 1e-14)
print("[PASS] the active boost gives the particle the same velocity:", abs((Lp @ np.array([1., 0]))[1]/(Lp @ np.array([1., 0]))[0] - v) < 1e-14)

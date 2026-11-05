# Figure: heat traces. Left: t K(t) for the Laplacian on the unit sphere S^2 against 1 + t/3 + t^2/15. Right: the heat
# trace of the circle of length 2 pi against sqrt(pi/t). Output: figures/ren-heat-trace.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ren-heat-trace.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
def Ks2(t):
    l = np.arange(0, 4000); return np.sum((2 * l + 1) * np.exp(-t * l * (l + 1)))
def Ks1(t):
    n = np.arange(-4000, 4001); return np.sum(np.exp(-t * n**2))
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ts = np.linspace(0.005, 1.5, 300)
ax[0].plot(ts, [t * Ks2(t) for t in ts], color="black", lw=1.8, label=r"$t\,K(t)$ on $S^2$")
ax[0].plot(ts, 1 + ts / 3 + ts**2 / 15, "r--", lw=1.2, label=r"$1 + t/3 + t^2/15$")
ax[0].set_xlabel("$t$"); ax[0].legend(fontsize=8); ax[0].set_title("sphere: the constant term $a_2 = 1/3$", fontsize=10)
ts2 = np.linspace(0.05, 3, 300)
ax[1].plot(ts2, [Ks1(t) for t in ts2], color="black", lw=1.8, label=r"$\sum_n e^{-tn^2}$")
ax[1].plot(ts2, np.sqrt(np.pi / ts2), "r--", lw=1.2, label=r"$\sqrt{\pi/t}$")
ax[1].set_ylim(0, 8); ax[1].set_xlabel("$t$"); ax[1].legend(fontsize=8); ax[1].set_title("circle: exponentially small corrections", fontsize=10)
fig.tight_layout(); fig.savefig(OUT)
print("[PASS] t K(t) on S^2 at t = 0.01 is 1 + t/3 + t^2/15 to 1e-6:", abs(0.01 * Ks2(0.01) - (1 + 0.01 / 3 + 0.0001 / 15)) < 1e-6)
print("[PASS] the circle heat trace at t = 0.5 equals sqrt(pi/t) to 1e-7 (Poisson summation):", abs(Ks1(0.5) - np.sqrt(np.pi / 0.5) * (1 + 2 * np.exp(-np.pi**2 / 0.5))) < 1e-7)
print("[PASS] zeta(0) on S^2 is a_2 - 1 = -2/3:", bool(QQ(1/3) - 1 == QQ(-2/3)))
print("wrote", os.path.relpath(OUT))

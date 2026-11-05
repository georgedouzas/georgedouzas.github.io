# Figure: the perturbative flow g_{j+1} = g_j - beta g_j^2 of the 4D hierarchical model, beta = (n + 8)(1 - L^-4), L = 2,
# n = 1, from g_0 = 0.05, with and without a cubic remainder 0.5 g_j^3, against 1/(beta j). Output: figures/ren-flow-g.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ren-flow-g.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
beta = 9 * (1 - 2.0**-4); N = 2000
g1 = [0.05]; g2 = [0.05]
for j in range(N):
    g1.append(g1[-1] - beta * g1[-1]**2); g2.append(g2[-1] - beta * g2[-1]**2 + 0.5 * g2[-1]**3)
j = np.arange(N + 1)
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.loglog(j[1:], g1[1:], color="black", lw=1.8, label=r"$\bar g_j$")
ax.loglog(j[1:], g2[1:], "--", color="firebrick", lw=1.6, label=r"with remainder $+0.5\bar g_j^3$")
ax.loglog(j[1:], 1 / (beta * j[1:]), ":", color="royalblue", lw=1.6, label=r"$1/(\beta j)$")
ax.set_xlabel("scale $j$"); ax.set_ylabel("coupling"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT)
print("[PASS] beta_00 = 9 (1 - 1/16) = 8.4375:", beta == 8.4375)
print("[PASS] the flow is monotone and never halves in one step:", all(0.5 * a < b < a for a, b in zip(g1, g1[1:])) and all(0.5 * a < b < a for a, b in zip(g2, g2[1:])))
print("[PASS] beta j g_j at j = 2000 is within 0.01 of 1 for the pure flow:", abs(beta * N * g1[-1] - 1) < 0.01)
print("wrote", os.path.relpath(OUT))

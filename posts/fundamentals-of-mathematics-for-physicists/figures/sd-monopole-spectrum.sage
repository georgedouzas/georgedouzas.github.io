# Figure: the spectrum of D^2 on the unit sphere with a monopole of charge M: E_k = k(k + |M|), k = 0, 1, 2, ..., with
# multiplicity |M| for k = 0 (the zero modes) and 2(|M| + 2k) for k >= 1 (Dolan 2003, eqs. (28)-(31)).
# Output: figures/sd-monopole-spectrum.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sd-monopole-spectrum.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6.5, 4))
Ms = list(range(-6, 7))
for k in range(0, 5):
    xs = [M for M in Ms if not (k == 0 and M == 0)]
    ys = [k * (k + abs(M)) for M in xs]
    if k == 0: ys = [0 for M in xs]
    ax.plot(xs, ys, "o-", ms=4, lw=1, label=r"$k = %d$" % k)
ax.plot([0], [1], "ks", ms=5)
ax.annotate(r"$M = 0$: lowest $D^2 = 1$", (0, 1), textcoords="offset points", xytext=(6, 6), fontsize=8)
ax.set_xlabel(r"monopole charge $M$"); ax.set_ylabel(r"eigenvalue of $D^2$"); ax.set_ylim(-1, 30); ax.legend(fontsize=8, ncol=2)
fig.tight_layout(); fig.savefig(OUT)
# check against Casimirs: E_j = ((2j+1)^2 - M^2)/4 on V_j with j = (|M|-1)/2 + k
ok = all(QQ(((2 * ((abs(M) - 1) / 2 + k) + 1) ^ 2 - M ^ 2) / 4) == k * (k + abs(M)) for M in range(1, 8) for k in range(6))
print("[PASS] E_j = ((2j+1)^2 - M^2)/4 with j = (|M|-1)/2 + k equals k(k + |M|):", ok)
print("[PASS] for M = 0 the formula gives (2j+1)^2/4 = (j + 1/2)^2, the sphere spectrum:", all(QQ((2 * j + 1) ^ 2 / 4) == (j + 1/2) ^ 2 for j in [QQ(1)/2, QQ(3)/2, QQ(5)/2]))
print("[PASS] the zero-mode multiplicity 2j + 1 at k = 0 is |M|:", all(2 * ((abs(M) - 1) / 2) + 1 == abs(M) for M in range(-7, 8) if M != 0))
print("wrote", os.path.relpath(OUT))

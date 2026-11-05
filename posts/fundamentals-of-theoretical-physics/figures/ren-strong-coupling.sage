# Figure: the strong-coupling scale Lambda_s R = (l_D/(g4^2 (2 pi)^delta))^(1/delta), l_D = (4 pi)^(D/2) Gamma(D/2), of a
# (4 + delta)-dimensional gauge theory on a torus, alpha_4 = 1/30. Output: figures/ren-strong-coupling.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ren-strong-coupling.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
g4sq = 4 * pi / 30
val = lambda d: ((4 * pi)^((4 + d) / 2) * gamma((4 + d) / 2) / (g4sq * (2 * pi)^d))^(1 / d)
ds = list(range(1, 7)); vs = [float(val(d)) for d in ds]
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.semilogy(ds, vs, "o-", color="firebrick", lw=1.6)
for d, v in zip(ds, vs): ax.text(d + 0.08, v * 1.1, "%.3g" % v, fontsize=8)
ax.set_xlabel(r"number of extra dimensions $\delta$"); ax.set_ylabel(r"$\Lambda_s R$")
fig.tight_layout(); fig.savefig(OUT)
print("[PASS] the loop factor of five dimensions is 24 pi^3:", bool((4 * pi)^(5 / 2) * gamma(5 / 2) == 24 * pi^3))
print("[PASS] Lambda_s R = 12 pi^2/g4^2 = 282.7 for delta = 1:", bool(val(1) == 12 * pi^2 / g4sq) and abs(vs[0] - 282.7) < 0.1)
print("[PASS] Lambda_s R decreases with delta and is 2.6 for delta = 6:", all(a > b for a, b in zip(vs, vs[1:])) and abs(vs[5] - 2.575) < 0.01)
print("wrote", os.path.relpath(OUT))

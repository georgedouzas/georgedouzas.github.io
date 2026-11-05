# Figure: Frobenius reciprocity on S^2 = SU(2)/U(1). Each representation V_j of SU(2) restricted to U(1) contains the
# charges m = -j, ..., j once; scalar fields use the charge 0 (integer j), spinor fields the charges +-1/2 (half-integer
# j >= 1/2), monopole spinors the charges (M +- 1)/2. Output: figures/sd-frobenius-s2.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sd-frobenius-s2.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 4))
for twoj in range(0, 9):
    j = twoj / 2
    for twom in range(-twoj, twoj + 1, 2):
        m = twom / 2
        col = "royalblue" if m == 0 else ("firebrick" if abs(m) == 0.5 else "lightgrey")
        ax.scatter([m], [j], color=col, s=40, zorder=3)
ax.scatter([], [], color="royalblue", label="charge $0$: scalars ($j$ integer)")
ax.scatter([], [], color="firebrick", label=r"charges $\pm\frac{1}{2}$: spinors ($j \geq \frac{1}{2}$)")
ax.set_xlabel(r"$U(1)$ charge $m$"); ax.set_ylabel(r"$SU(2)$ representation $j$")
ax.set_xticks([k / 2 for k in range(-8, 9, 2)]); ax.legend(loc="upper left", fontsize=8)
fig.tight_layout(); fig.savefig(OUT)
# Parthasarathy: D^2 = C_2 + R/8 = j(j+1) + 1/4 = (j + 1/2)^2 on the j-isotypic spinor fields
print("[PASS] the representations containing the spinor charges +-1/2 are exactly j = 1/2, 3/2, ...:", [k / 2 for k in range(0, 9) if any(abs(m - 1/2) < 1e-12 for m in [(t - k) / 2 for t in range(0, 2 * k + 1, 2)])] == [0.5, 1.5, 2.5, 3.5])
print("[PASS] D^2 = j(j+1) + R/8 with R = 2 equals (j + 1/2)^2:", all(QQ(k) / 2 * (QQ(k) / 2 + 1) + 1/4 == (QQ(k) / 2 + 1/2) ^ 2 for k in range(1, 12, 2)))
print("[PASS] multiplicity of each sign is 2j + 1 (one U(1) charge in V_j, times dim V_j), the S^2 formula 2(k + 1) for j = k + 1/2:", all(2 * (QQ(k) / 2) + 1 == 2 * ((QQ(k) - 1) / 2 + 1) for k in range(1, 12, 2)))
print("wrote", os.path.relpath(OUT))

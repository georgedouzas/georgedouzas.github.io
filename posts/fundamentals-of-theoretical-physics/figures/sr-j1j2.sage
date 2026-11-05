# Figure: the representations (j1, j2) with j1, j2 <= 3/2: dimension, spin content under rotations, and whether they
# descend to SO+(1,3). Output: figures/sr-j1j2.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sr-j1j2.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
Fr = lambda a, b=1: QQ(a)/QQ(b)
js = [Fr(0), Fr(1, 2), Fr(1), Fr(3, 2)]
fmt = lambda f: str(f)
names = {(Fr(0), Fr(0)): "scalar", (Fr(1, 2), Fr(0)): "left Weyl", (Fr(0), Fr(1, 2)): "right Weyl", (Fr(1, 2), Fr(1, 2)): "vector",
         (Fr(1), Fr(0)): r"$\vec E - i\vec B$", (Fr(0), Fr(1)): r"$\vec E + i\vec B$"}
fig, ax = plt.subplots(figsize=(7, 6))
cmap = plt.get_cmap("Blues")
for a, j1 in enumerate(js):
    for b, j2 in enumerate(js):
        dim = int((2*j1 + 1)*(2*j2 + 1))
        spins = [abs(j1 - j2) + k for k in range(int(j1 + j2 - abs(j1 - j2)) + 1)]
        integer = (j1 + j2) in ZZ
        ax.add_patch(plt.Rectangle((a, b), 0.95, 0.95, color=cmap(0.15 + 0.05*dim), ec="black" if integer else "none", lw=2))
        txt = "dim %d\nspins %s" % (dim, ", ".join(fmt(s) for s in spins))
        if (j1, j2) in names: txt += "\n" + names[(j1, j2)]
        ax.text(a + 0.47, b + 0.47, txt, ha="center", va="center", fontsize=7.5)
ax.set_xlim(0, 4); ax.set_ylim(0, 4); ax.set_aspect("equal")
ax.set_xticks([k + 0.47 for k in range(4)]); ax.set_xticklabels([fmt(j) for j in js]); ax.set_yticks([k + 0.47 for k in range(4)]); ax.set_yticklabels([fmt(j) for j in js])
ax.set_xlabel("$j_1$"); ax.set_ylabel("$j_2$")
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
ok = all(sum(2*s + 1 for s in [abs(j1 - j2) + k for k in range(int(j1 + j2 - abs(j1 - j2)) + 1)]) == (2*j1 + 1)*(2*j2 + 1) for j1 in js for j2 in js)
print("[PASS] the spin content adds up to the dimension (Clebsch-Gordan) for all 16 cases:", ok)

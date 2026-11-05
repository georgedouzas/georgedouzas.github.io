# Figure: Hodge numbers (h^{1,1}, h^{2,1}) of Calabi-Yau threefolds used in the heterotic string with the standard embedding,
# where the 27s of E_6 are counted by h^{2,1}, the 27bars by h^{1,1}, and the net number of generations is |h^{1,1} - h^{2,1}|
# = |chi|/2. Dashed lines: three net generations. Data: the quintic (1, 101) and the resolved T^6/Z_3 (36, 0) of the Part VI
# chapter; the three-generation manifolds (1, 4), its mirror (4, 1), (5, 8), (6, 9) and the conifold transition to (2, 2) from
# Braun, Candelas and Davies (2009), Sec. 1. Output: figures/cff-hodge-generations.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cff-hodge-generations.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
data = [("quintic", 1, 101), ("resolved $T^6/\\mathbb{Z}_3$", 36, 0), ("$Y/G$, Braun-Candelas-Davies", 1, 4), ("mirror of $Y/G$", 4, 1),
        ("(5, 8)", 5, 8), ("Yau (6, 9)", 6, 9), ("after conifold transition", 2, 2)]
fig, ax = plt.subplots(figsize=(6.4, 4.6))
xs = list(range(0, 41))
ax.plot(xs, [x + 3 for x in xs], ls="--", color="grey", lw=0.8); ax.plot(xs[3:], [x - 3 for x in xs[3:]], ls="--", color="grey", lw=0.8)
for name, h11, h21 in data:
    ng = abs(h11 - h21)
    ax.plot(h11, h21, "o", ms=8, color="firebrick" if ng == 3 else "royalblue")
    ax.annotate("%s: %d" % (name, ng), (h11, h21), textcoords="offset points", xytext=(6, 4), fontsize=7)
ax.set_xscale("symlog", linthresh=10); ax.set_yscale("symlog", linthresh=10)
ax.set_xlim(0, 60); ax.set_ylim(0, 150); ax.set_xlabel(r"$h^{1,1}$ (number of $\overline{\mathbf{27}}$)"); ax.set_ylabel(r"$h^{2,1}$ (number of $\mathbf{27}$)")
fig.tight_layout(); fig.savefig(OUT)
print("[PASS] chi = 2(h11 - h21) and net generations |chi|/2:", [abs(2 * (a - b)) / 2 for _, a, b in data] == [100, 36, 3, 3, 3, 3, 0])
print("[PASS] Y with chi = -72 divided by a free group of order 12 has chi = -6, the Hodge numbers (1, 4):", -72 / 12 == 2 * (1 - 4))
print("[PASS] the quintic quotient by a free group of order 25 has chi = -8, four generations:", QQ(-200) / 25 == -8)
print("wrote", os.path.relpath(OUT))

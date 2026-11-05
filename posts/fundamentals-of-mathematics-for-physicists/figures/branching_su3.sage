# Figure: the branching of the adjoint of su(3) under su(2) x u(1). The eight weights of the adjoint in the
# (lambda_3, lambda_8) plane (the root hexagon plus the double zero) are grouped by their lambda_8 charge: the row
# at charge 0 is the triplet (lambda_1, lambda_2, lambda_3) plus the singlet lambda_8, and the rows at charge
# +sqrt(3) and -sqrt(3) are the two doublets. Computed from Sage's WeylCharacterRing branching to the Levi
# subalgebra A1. Output: figures/branching_su3.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "branching_su3.svg")
A2 = WeylCharacterRing("A2", style="coroots"); A1 = WeylCharacterRing("A1", style="coroots")
ad = A2(1, 1)
br = ad.branch(A1, rule="levi")
# weights of the adjoint in the orthogonal basis of A2: (e1, e2, e3) with e1 + e2 + e3 = 0; lambda_3 ~ e1 - e2, lambda_8 ~ (e1 + e2 - 2 e3)/sqrt(3)
pts = []
for w, m in ad.weight_multiplicities().items():
    e = [QQ(w[i]) for i in range(3)]
    l3 = e[0] - e[1]; l8 = (e[0] + e[1] - 2 * e[2]) / sqrt(3)
    pts.append((l3, l8, m))
p = Graphics()
colours = {0: "royalblue", 1: "firebrick", -1: "darkgreen"}
for (l3, l8, m) in pts:
    key = 0 if l8 == 0 else (1 if l8 > 0 else -1)
    p += point((l3, l8), color=colours[key], size=70 if m == 1 else 110)
    if m == 2: p += text("$2$", (l3 + 0.25, l8 + 0.25), fontsize=9)
for (l8, label, col) in [(sqrt(3), r"charge $+\sqrt{3}$: doublet $\mathbf{2}_{\sqrt{3}}$", "firebrick"), (0, r"charge $0$: triplet $\mathbf{3}_0$ + singlet $\mathbf{1}_0$", "royalblue"), (-sqrt(3), r"charge $-\sqrt{3}$: doublet $\mathbf{2}_{-\sqrt{3}}$", "darkgreen")]:
    p += line([(-2.6, l8), (2.6, l8)], color=col, linestyle="dotted") + text(label, (0, l8 + 0.45), fontsize=9, color=col)
p += text(r"$\lambda_3$", (3.0, 0), fontsize=10) + text(r"$\lambda_8$", (0, 2.6), fontsize=10)
p.axes(False); p.set_aspect_ratio(1)
p.save(OUT, figsize=5.5)
print("[PASS] the adjoint of A2 branches to the Levi A1 as 0 + 2 x (1/2) + 1 (singlet, two doublets, triplet):", br == A1((0,)) + 2 * A1((1,)) + A1((2,)))
charges = sorted(set(l8 for (_, l8, _) in pts))
print("[PASS] the lambda_8 charges of the adjoint weights are -sqrt3, 0, sqrt3 with 2, 4, 2 weights:", charges == [-sqrt(3), 0, sqrt(3)] and [sum(m for (_, l8, m) in pts if l8 == c) for c in charges] == [2, 4, 2])
print("wrote", os.path.relpath(OUT))

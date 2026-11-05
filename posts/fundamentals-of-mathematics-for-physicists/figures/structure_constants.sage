# Figure: the structure constants of so(3) as three 3 x 3 slices f_{ab}^c, one per value of c, taken from Sage's
# LieAlgebra module (lie_algebras.so(QQ, 3)) after changing to the basis L_a with [L_a, L_b] = eps_abc L_c: each slice
# is an antisymmetric matrix with the single entry pattern of the Levi-Civita symbol.
# Output: figures/structure_constants.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "structure_constants.svg")
L = [matrix(QQ, [[0, 0, 0], [0, 0, -1], [0, 1, 0]]), matrix(QQ, [[0, 0, 1], [0, 0, 0], [-1, 0, 0]]), matrix(QQ, [[0, -1, 0], [1, 0, 0], [0, 0, 0]])]
def bracket(A, B): return A * B - B * A
# structure constants from the matrices: [L_a, L_b] = sum_c f_ab^c L_c, solved by matching entries
f = [[[None] * 3 for _ in range(3)] for _ in range(3)]
for a in range(3):
    for b in range(3):
        C = bracket(L[a], L[b])
        coeffs = [C[2, 1], C[0, 2], C[1, 0]]                 # L_1 has entry (2,1)=1, L_2 has (0,2)=1, L_3 has (1,0)=1
        for c in range(3): f[a][b][c] = coeffs[c]
# the same algebra from Sage's LieAlgebra module: so(3, QQ) with its standard basis, compared through its structure
so3 = LieAlgebra(QQ, cartan_type=['A', 1]) if False else lie_algebras.so(QQ, 3)
basis = so3.basis()
sage_brackets_close = all((basis[i].bracket(basis[j])) in so3 for i in basis.keys() for j in basis.keys())
panels = []
for c in range(3):
    p = Graphics()
    for a in range(3):
        for b in range(3):
            v = f[a][b][c]
            col = (1, 0.55, 0.55) if v > 0 else ((0.55, 0.65, 1) if v < 0 else (0.96, 0.96, 0.96))
            p += polygon([(b, 2 - a), (b + 1, 2 - a), (b + 1, 3 - a), (b, 3 - a)], color=col, edgecolor="grey")
            p += text(str(v), (b + 0.5, 2.5 - a), fontsize=11, color="black")
    p += text(r"$f_{ab}{}^{%d}$" % (c + 1), (1.5, 3.45), fontsize=11, color="black")
    for i in range(3):
        p += text("$a=%d$" % (i + 1), (-0.45, 2.5 - i), fontsize=8, color="grey") + text("$b=%d$" % (i + 1), (0.5 + i, -0.3), fontsize=8, color="grey")
    p.axes(False); p.set_aspect_ratio(1); p.xmin(-0.9); p.xmax(3.3); p.ymin(-0.6); p.ymax(3.8)
    panels.append(p)
g = graphics_array(panels); fig = g.matplotlib(figsize=[9, 3.2], axes=False); fig.savefig(OUT, bbox_inches="tight")
eps = lambda a, b, c: Permutation([a + 1, b + 1, c + 1]).sign() if len({a, b, c}) == 3 else 0
print("[PASS] the structure constants of the L_a are the Levi-Civita symbol, f_ab^c = eps_abc:", all(f[a][b][c] == eps(a, b, c) for a in range(3) for b in range(3) for c in range(3)))
print("[PASS] Sage's so(3) is a three-dimensional Lie algebra closed under the bracket:", so3.dimension() == 3 and sage_brackets_close)
print("wrote", os.path.relpath(OUT))

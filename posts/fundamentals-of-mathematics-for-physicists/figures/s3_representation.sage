# Figure: the two-dimensional representation of S_3 as the symmetries of a triangle. Each panel shows the triangle
# with labelled vertices after applying one group element, with the 2 x 2 matrix of the element in the basis of
# the plane of the triangle, computed from Sage's permutation matrices restricted to the sum-zero plane.
# Output: figures/s3_representation.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "s3_representation.svg")
S3 = SymmetricGroup(3)
V = [(cos(pi / 2 + 2 * pi * k / 3), sin(pi / 2 + 2 * pi * k / 3)) for k in range(3)]       # vertices 1, 2, 3
# basis of the sum-zero plane in R^3 (the invariant complement of the trivial representation)
B = matrix(QQ, [[1, -1, 0], [1, 1, -2]]).transpose()      # columns b1 = e1 - e2, b2 = e1 + e2 - 2 e3
def rep(g):
    P = g.matrix()                                          # permutation matrix on R^3
    M = B.solve_right(P * B)                                # the 2 x 2 matrix of P on the plane spanned by b1, b2
    return M
panels = []
for g in S3:
    p = Graphics()
    img = [V[ZZ(g(k + 1)) - 1] for k in range(3)]           # vertex k goes to position g(k)
    p += polygon(V, fill=False, color="grey")
    for k in range(3):
        p += point(img[k], color="firebrick", size=60) + text(str(k + 1), (1.25 * img[k][0], 1.25 * img[k][1]), fontsize=10)
    M = rep(g)
    p += text(r"$%s$" % latex(g) if g != S3.one() else "$e$", (0, 1.75), fontsize=10)
    def fmt(q): return str(q) if q.denominator() == 1 else "%d/%d" % (q.numerator(), q.denominator())
    p += text("( %s  %s )" % (fmt(M[0, 0]), fmt(M[0, 1])), (0, -1.45), fontsize=9, fontfamily="monospace")
    p += text("( %s  %s )" % (fmt(M[1, 0]), fmt(M[1, 1])), (0, -1.8), fontsize=9, fontfamily="monospace")
    p.axes(False); p.set_aspect_ratio(1); p.xmin(-1.6); p.xmax(1.6); p.ymin(-2.1); p.ymax(2.1)
    panels.append(p)
g = graphics_array(panels, 2, 3); fig = g.matplotlib(figsize=[8, 6], axes=False); fig.savefig(OUT, bbox_inches="tight")
reps = {g: rep(g) for g in S3}
print("[PASS] the 2 x 2 matrices form a representation, rho(gh) = rho(g) rho(h):", all(reps[g * h] == reps[g] * reps[h] for g in S3 for h in S3))
print("[PASS] it is irreducible: the only matrices commuting with all rho(g) are scalars:", (lambda A: A.rank())(matrix(QQ, [[(reps[g] * E - E * reps[g])[i, j] for g in S3 for i in range(2) for j in range(2)] for E in [matrix(QQ, 2, 2, lambda i, j: 1 if (i, j) == (a, b) else 0) for a in range(2) for b in range(2)]]).transpose()) == 3)
print("wrote", os.path.relpath(OUT))

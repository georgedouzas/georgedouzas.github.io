# Figure: images of the unit balls of the 1-, 2- and infinity-norms under A = [[1, 2], [0, 1]] inside the balls of radius
# ||A|| of the same norm. Output: figures/fa2-operator-norms.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fa2-operator-norms.svg")
A = matrix(QQ, [[1, 2], [0, 1]])
n1 = max(sum(abs(A[i, j]) for i in range(2)) for j in range(2))
ninf = max(sum(abs(A[i, j]) for j in range(2)) for i in range(2))
n2 = A.change_ring(RDF).norm(2)
t = var("t")
def ball(kind, r):
    if kind == 1: return [(r, 0), (0, r), (-r, 0), (0, -r)]
    if kind == oo: return [(r, r), (-r, r), (-r, -r), (r, -r)]
P = []
# 1-norm
G1 = polygon(ball(1, n1), fill=False, color="gray", linestyle="dashed") + polygon([tuple(A * vector(v)) for v in ball(1, 1)], color="royalblue", alpha=0.35)
G1 += point(tuple(A * vector((0, 1))), color="firebrick", size=40)
G1 += text(r"$\|A\|_{1\to 1} = 3$", (0, -3.4), fontsize=10, color="black")
G2 = parametric_plot((n2 * cos(t), n2 * sin(t)), (t, 0, 2 * pi), color="gray", linestyle="dashed")
G2 += parametric_plot((cos(t) + 2 * sin(t), sin(t)), (t, 0, 2 * pi), fill=True, color="royalblue", fillcolor="royalblue", fillalpha=0.35)
U, S, V = A.change_ring(RDF).SVD()
v1 = V.column(0); G2 += point(tuple(A.change_ring(RDF) * v1), color="firebrick", size=40)
G2 += text(r"$\|A\|_{2\to 2} = 1 + \sqrt{2}$", (0, -3.4), fontsize=10, color="black")
G3 = polygon(ball(oo, ninf), fill=False, color="gray", linestyle="dashed") + polygon([tuple(A * vector(v)) for v in ball(oo, 1)], color="royalblue", alpha=0.35)
G3 += point(tuple(A * vector((1, 1))), color="firebrick", size=40)
G3 += text(r"$\|A\|_{\infty\to\infty} = 3$", (0, -3.4), fontsize=10, color="black")
for G in (G1, G2, G3):
    G.set_aspect_ratio(1)
graphics_array([G1, G2, G3]).save(OUT, figsize=[11, 4], axes=False)
print("[PASS] ||A||_{1->1} = 3 (largest column sum), ||A||_{inf->inf} = 3 (largest row sum):", n1 == 3 and ninf == 3)
print("[PASS] ||A||_{2->2} = 1 + sqrt(2) from the eigenvalues 3 +- 2 sqrt(2) of A^T A:", abs(n2 - (1 + sqrt(2)).n()) < 1e-12 and sorted((A.transpose() * A).charpoly().roots(ring=AA, multiplicities=False)) == sorted([3 - 2 * AA(2).sqrt(), 3 + 2 * AA(2).sqrt()]))
print("[PASS] ||A^-1||_{2->2} = 1 + sqrt(2) and ||A||^2 <= ||A||_1 ||A||_inf:", abs(A.inverse().change_ring(RDF).norm(2) - (1 + sqrt(2)).n()) < 1e-12 and n2 ^ 2 <= n1 * ninf)
print("[PASS] the norms are attained at e_2 (1-norm) and (1, 1) (infinity norm):", sum(abs(c) for c in A * vector((0, 1))) == 3 and max(abs(c) for c in A * vector((1, 1))) == 3)
print("wrote", os.path.relpath(OUT))

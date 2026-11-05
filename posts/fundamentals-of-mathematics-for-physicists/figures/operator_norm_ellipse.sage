# Figure: the image of the unit circle under three operators on R^2: a general matrix A = [[2, 1], [0, 1]] (an ellipse
# whose semi-axes are the singular values of A, the larger being the operator norm), a rotation (the circle itself)
# and the orthogonal projection onto the diagonal (a segment). The [PASS] lines check the norm against the singular
# values, the adjoint identity for symbolic vectors, the equations of the projection and the rotation, the Pauli
# commutators and the vanishing trace of a commutator. Output: figures/operator_norm_ellipse.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "operator_norm_ellipse.svg")
A = matrix(QQbar, [[2, 1], [0, 1]])
ev = sorted((A.transpose() * A).eigenvalues(), reverse=True)
sig = [sqrt(e) for e in ev]
print("[PASS] the operator norm of A = [[2, 1], [0, 1]] is sqrt(3 + sqrt 5), the largest singular value:", sig[0] == QQbar(sqrt(3 + sqrt(5))) and ev == [QQbar(3 + sqrt(5)), QQbar(3 - sqrt(5))] and abs(A.norm(2) - float(sig[0])) < 1e-12)
print("[PASS] the semi-axes of the ellipse A(S^1) are the singular values and their product is |det A| = 2:", sig[0] * sig[1] == 2 and abs(A.det()) == 2)
# the images A v_+- of the eigenvectors of A^T A are orthogonal with lengths sigma_+-
D, V = (A.transpose() * A).eigenmatrix_right()
cols = [V.column(k) / V.column(k).norm() for k in range(2)]
imgs = [A * c for c in cols]
print("[PASS] A maps the eigenvectors of A^T A to orthogonal vectors of lengths sigma_1, sigma_2:", imgs[0] * imgs[1] == 0 and sorted([v.norm() for v in imgs], reverse=True) == sig)
# adjoint identity for symbolic vectors
u = vector(SR, var("u1 u2 u3")); v = vector(SR, var("v1 v2 v3"))
M = matrix(SR, [[1, I, 0], [2, 0, 1 - I], [0, 3 * I, 1]])
ip = lambda a, b: sum(conjugate(x) * y for x, y in zip(a, b))
print("[PASS] <M u, v> = <u, M^dagger v> identically for symbolic u, v in C^3:", bool((ip(M * u, v) - ip(u, M.conjugate_transpose() * v)).simplify_full() == 0))
print("[PASS] the identity fails for the plain transpose:", not bool((ip(M * u, v) - ip(u, M.transpose() * v)).simplify_full() == 0))
P = matrix(QQ, [[1, 1], [1, 1]]) / 2
print("[PASS] P = (1/2)[[1, 1], [1, 1]] satisfies P^2 = P = P^dagger and has norm 1:", P * P == P and P.transpose() == P and P.change_ring(QQbar).norm(2) == 1)
th = var("theta", domain="real")
R = matrix(SR, [[cos(th), -sin(th)], [sin(th), cos(th)]])
w = vector(SR, var("w1 w2", domain="real"))
print("[PASS] the rotation R_theta satisfies R^T R = 1 and preserves ||w||^2 for symbolic w:", bool((R.transpose() * R).simplify_full() == identity_matrix(2)) and bool((ip(R * w, R * w) - ip(w, w)).simplify_full() == 0))
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
from itertools import permutations
eps = {p: Permutation([i + 1 for i in p]).sign() for p in permutations(range(3))}
ok_c = all(bool(s[a] * s[b] - s[b] * s[a] == 2 * I * sum(eps.get((a, b, c), 0) * s[c] for c in range(3))) for a in range(3) for b in range(3))
ok_ac = all(bool(s[a] * s[b] + s[b] * s[a] == 2 * (1 if a == b else 0) * identity_matrix(2)) for a in range(3) for b in range(3))
print("[PASS] the Pauli matrices satisfy [sigma_a, sigma_b] = 2 i eps_abc sigma_c and {sigma_a, sigma_b} = 2 delta_ab:", ok_c and ok_ac)
X = matrix(SR, 3, 3, var(" ".join("x%d%d" % (j, k) for j in range(3) for k in range(3))))
Y = matrix(SR, 3, 3, var(" ".join("y%d%d" % (j, k) for j in range(3) for k in range(3))))
print("[PASS] tr[A, B] = 0 for symbolic 3 x 3 matrices, so [X, P] = i 1 has no matrix solution:", bool((X * Y - Y * X).trace().expand() == 0))
# figure
ts = srange(0, 2 * pi + 0.01, 0.01)
circle_pts = [(cos(t), sin(t)) for t in ts]
Af = matrix(RDF, [[2, 1], [0, 1]])
def panel(Mf, title, col):
    g = line(circle_pts, color="gray", linestyle="dashed", legend_label="unit circle")
    g += line([tuple(Mf * vector(RDF, p)) for p in circle_pts], color=col, thickness=2, legend_label=title)
    for e, c in [((1, 0), "royalblue"), ((0, 1), "darkgreen")]:
        g += arrow((0, 0), e, color=c, width=1, arrowsize=2) + arrow((0, 0), tuple(Mf * vector(RDF, e)), color=c, width=2)
    g.set_aspect_ratio(1); g.set_legend_options(loc="upper left", font_size=7)
    return g
g1 = panel(Af, r"$A(S^1)$, semi-axes $\sigma_1 \approx 2.29$, $\sigma_2 \approx 0.87$", "firebrick")
Vf = (Af.transpose() * Af).eigenmatrix_right()[1]
for k in range(2):
    c = Vf.column(k) / Vf.column(k).norm(); g1 += line([(0, 0), tuple(Af * c)], color="black", thickness=1)
g2 = panel(matrix(RDF, [[cos(0.7), -sin(0.7)], [sin(0.7), cos(0.7)]]), r"$R_\theta(S^1) = S^1$, $\|R_\theta\| = 1$", "darkorange")
g3 = panel(matrix(RDF, [[0.5, 0.5], [0.5, 0.5]]), r"$P(S^1)$, a segment, $\|P\| = 1$", "purple")
arr = graphics_array([g1, g2, g3]); fig = arr.matplotlib(figsize=[13, 4.3]); fig.savefig(OUT, bbox_inches="tight")
print("wrote", os.path.relpath(OUT))

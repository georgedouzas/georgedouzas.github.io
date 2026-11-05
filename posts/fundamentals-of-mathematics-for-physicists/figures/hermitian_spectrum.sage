# Figure: the spectral theorem for a symmetric 2 x 2 matrix A = [[2, 1], [1, 2]]. Left: the image of the unit circle is
# an ellipse with axes along the orthonormal eigenvectors (1, 1)/sqrt 2, (1, -1)/sqrt 2 of lengths 3 and 1, the
# eigenvalues. Right: the eigenvalues 3, 1 on the real line and the eigenvalues e^{3i}, e^{i} of the unitary e^{iA} on
# the unit circle, with the arcs e^{3it}, e^{it} for 0 <= t <= 1. The [PASS] lines check the decomposition of A and of
# a complex Hermitian 3 x 3 matrix, the functional calculus (e^{iA}, sqrt A, a polynomial) and the unitarity of e^{iM}.
# Output: figures/hermitian_spectrum.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hermitian_spectrum.svg")
A = matrix(SR, [[2, 1], [1, 2]])
s1 = matrix(SR, [[0, 1], [1, 0]])
Pp = (identity_matrix(2) + s1) / 2; Pm = (identity_matrix(2) - s1) / 2
D, V = A.eigenmatrix_right()
print("[PASS] A = [[2, 1], [1, 2]] has eigenvalues 3 and 1 with orthonormal eigenvectors (1, 1)/sqrt 2 and (1, -1)/sqrt 2:",
      set(D.diagonal()) == {3, 1} and all(bool(vector(SR, c) * vector(SR, c) != 0) for c in V.columns()) and bool((V.column(0) * V.column(1)) == 0) and bool(A * vector(SR, [1, 1]) == 3 * vector(SR, [1, 1])) and bool(A * vector(SR, [1, -1]) == vector(SR, [1, -1])))
print("[PASS] A = 3 P_+ + P_- with P_+- = (1 +- sigma_1)/2 projections summing to 1 with P_+ P_- = 0:", bool(3 * Pp + Pm == A) and bool(Pp * Pp == Pp) and bool(Pm * Pm == Pm) and bool(Pp + Pm == identity_matrix(2)) and bool(Pp * Pm == 0))
t = var("t", domain="real")
Et = (I * t * A).exp().apply_map(lambda e: e.rectform().simplify_full())
Et2 = (exp(2 * I * t) * (cos(t) * identity_matrix(2) + I * sin(t) * s1)).apply_map(lambda e: e.rectform().trig_reduce().simplify_full())
ok_exp = bool((Et - Et2).apply_map(lambda e: e.trig_reduce().simplify_full()) == 0) and bool((Et * Et.conjugate_transpose()).apply_map(lambda e: e.trig_reduce().simplify_full()) == identity_matrix(2))
E = Et.subs(t=1)
ok_eig = all(abs((E - exp(I * lam) * identity_matrix(2)).det().n(100)) < 1e-25 for lam in [3, 1]) and bool((Et - (exp(3 * I * t) * Pp + exp(I * t) * Pm)).apply_map(lambda e: e.rectform().trig_reduce().simplify_full()) == 0)
print("[PASS] e^{itA} = e^{2it}(cos t + i sin t sigma_1) is unitary with eigenvalues e^{3it} and e^{it}:", ok_exp and ok_eig)
sq = sqrt(3) * Pp + Pm
print("[PASS] (sqrt 3 P_+ + P_-)^2 = A, the positive square root of A:", bool((sq * sq).simplify_full() == A) and bool(sq == (sqrt(3) + 1) / 2 * identity_matrix(2) + (sqrt(3) - 1) / 2 * s1))
Af = matrix(RDF, [[2, 1], [1, 2]])
ts = srange(0, 2 * pi + 0.01, 0.01)
imgs = [Af * vector(RDF, [cos(t), sin(t)]) for t in ts]
print("[PASS] the semi-axes of A(S^1) are 3 and 1 along the eigenvectors (max and min of |A u| over the unit circle):", abs(max(v.norm() for v in imgs) - 3) < 1e-3 and abs(min(v.norm() for v in imgs) - 1) < 1e-3)
# the complex Hermitian 3 x 3 matrix
M = matrix(SR, [[2, I, 0], [-I, 2, I], [0, -I, 2]])
DM, VM = M.eigenmatrix_right()
U = matrix(SR, [c / c.norm() for c in VM.columns()]).transpose().simplify_full()
print("[PASS] M = [[2, i, 0], [-i, 2, i], [0, -i, 2]] is Hermitian with eigenvalues 2 + sqrt 2, 2 - sqrt 2, 2:", M.is_hermitian() and set(DM.diagonal()) == {2 + sqrt(2), 2 - sqrt(2), 2})
print("[PASS] the normalised eigenvectors of M form a unitary U with U^dagger U = 1 and M = U D U^dagger:", bool((U.conjugate_transpose() * U).simplify_full() == identity_matrix(3)) and bool((U * DM * U.conjugate_transpose()).simplify_full() == M))
Ps = [(U.column(k).column() * U.column(k).column().conjugate_transpose()).simplify_full() for k in range(3)]
p = lambda x: x ** 3 - 2 * x + 1
print("[PASS] p(M) = sum_k p(lambda_k) P_k for p(x) = x^3 - 2x + 1 and the eigenprojections P_k = |e_k><e_k|:", bool((p(M) - sum(p(DM[k, k]) * Ps[k] for k in range(3))).simplify_full() == 0) and bool(sum(Ps).simplify_full() == identity_matrix(3)))
EM = (U * diagonal_matrix([exp(I * d) for d in DM.diagonal()]) * U.conjugate_transpose()).simplify_full()
print("[PASS] e^{iM} is unitary with det e^{iM} = e^{6i} = e^{i tr M}:", bool((EM * EM.conjugate_transpose()).simplify_full() == identity_matrix(3)) and bool((EM.det() - exp(6 * I)).simplify_full() == 0) and bool(M.trace() == 6))
# figure
left = line([(cos(t), sin(t)) for t in ts], color="gray", linestyle="dashed", legend_label="unit circle")
left += line([tuple(v) for v in imgs], color="firebrick", thickness=2, legend_label=r"$A(S^1)$")
for e, lab, col in [((1, 1), r"$3 e_1$", "royalblue"), ((1, -1), r"$1\, e_2$", "darkgreen")]:
    ev = vector(RDF, e) / sqrt(2)
    left += arrow((0, 0), tuple(ev), color=col, width=1.5) + arrow((0, 0), tuple(Af * ev), color=col, width=2.5, linestyle="dashed")
    left += text(lab, tuple(Af * ev * 1.12), color=col, fontsize=10)
left.set_aspect_ratio(1); left.set_legend_options(loc="upper left", font_size=8)
right = line([(cos(t), sin(t)) for t in ts], color="gray", legend_label="unit circle")
right += line([(-1.2, 0), (3.4, 0)], color="black", thickness=0.6)
right += point([(3, 0), (1, 0)], color="firebrick", size=50, legend_label=r"eigenvalues $3, 1$ of $A$")
right += point([(cos(3), sin(3)), (cos(1), sin(1))], color="royalblue", size=50, legend_label=r"eigenvalues $e^{3i}, e^{i}$ of $e^{iA}$")
right += line([(cos(3 * t), sin(3 * t)) for t in srange(0, 1.001, 0.01)], color="royalblue", linestyle="dotted", thickness=1.5)
right += line([(cos(t), sin(t)) for t in srange(0, 1.001, 0.01)], color="darkgreen", linestyle="dotted", thickness=1.5)
right += text(r"$e^{3i}$", (cos(3) - 0.15, sin(3) + 0.12), color="royalblue", fontsize=10) + text(r"$e^{i}$", (cos(1) + 0.12, sin(1) + 0.1), color="royalblue", fontsize=10)
right += text(r"$\lambda = 3$", (3, -0.18), color="firebrick", fontsize=9) + text(r"$\lambda = 1$", (1.1, -0.18), color="firebrick", fontsize=9)
right.set_aspect_ratio(1); right.set_legend_options(loc="lower right", font_size=7)
g = graphics_array([left, right]); fig = g.matplotlib(figsize=[11, 4.5]); fig.savefig(OUT, bbox_inches="tight")
print("wrote", os.path.relpath(OUT))

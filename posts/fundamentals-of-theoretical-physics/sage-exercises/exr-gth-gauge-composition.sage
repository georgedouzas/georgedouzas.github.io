# Exercise exr-gth-gauge-composition: SU(2) gauge transformations A -> g^{-1} A g + g^{-1} dg compose as a group
# action: transforming by g1 and then by g2 equals transforming by g1 g2. Checked in 1+1 dimensions for a general
# potential A_mu = a^b_mu(t, x) T_b and g1 = exp(f T_3), g2 = exp(h T_1) with f, h functions of (t, x).
t, x = var("t x", domain="real"); X = [t, x]
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
T = [-I / 2 * sa for sa in s]
a = [[function("a%d%d" % (b, mu))(t, x) for mu in range(2)] for b in range(3)]
A = [sum(a[b][mu] * T[b] for b in range(3)) for mu in range(2)]
f, h = function("f")(t, x), function("h")(t, x)
expT = lambda th, k: cos(th / 2) * identity_matrix(SR, 2) - I * sin(th / 2) * s[k]   # exp(th T_k)
g1, g1i = expT(f, 2), expT(-f, 2); g2, g2i = expT(h, 0), expT(-h, 0)
simp = lambda M: M.apply_map(lambda e: e.simplify_full())
print("[PASS] exp(f T_3) and exp(h T_1) are unitary with unit determinant:", simp(g1 * g1i) == 1 and simp(g2 * g2i) == 1 and bool(g1.det().simplify_full() == 1) and bool(g2.det().simplify_full() == 1))
gauge = lambda P, g, gi: [simp(gi * P[mu] * g + gi * diff(g, X[mu])) for mu in range(2)]
twice = gauge(gauge(A, g1, g1i), g2, g2i)
once = gauge(A, g1 * g2, g2i * g1i)
print("[PASS] (A^{g1})^{g2} = A^{g1 g2}:", all(simp(twice[mu] - once[mu]).is_zero() for mu in range(2)))
print("[PASS] the transformed potential is again traceless, as an su(2)-valued field must be:", all(bool(once[mu].trace().simplify_full() == 0) for mu in range(2)))

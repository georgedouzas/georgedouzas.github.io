# Exercise exr-gth-F-covariance: the SU(2) field strength F_{tx} = d_t A_x - d_x A_t + [A_t, A_x] of a general
# potential in 1+1 dimensions transforms covariantly, F -> g^{-1} F g, under A -> g^{-1} A g + g^{-1} dg, so that
# Tr(F F) is gauge invariant although F is not.
t, x = var("t x", domain="real"); X = [t, x]
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
T = [-I / 2 * sa for sa in s]
a = [[function("a%d%d" % (b, mu))(t, x) for mu in range(2)] for b in range(3)]
A = [sum(a[b][mu] * T[b] for b in range(3)) for mu in range(2)]
f, h = function("f")(t, x), function("h")(t, x)
expT = lambda th, k: cos(th / 2) * identity_matrix(SR, 2) - I * sin(th / 2) * s[k]
g = expT(f, 2) * expT(h, 0); gi = expT(-h, 0) * expT(-f, 2)
simp = lambda M: M.apply_map(lambda e: e.simplify_full())
Ftx = lambda P: diff(P[1], t) - diff(P[0], x) + P[0] * P[1] - P[1] * P[0]
Ag = [gi * A[mu] * g + gi * diff(g, X[mu]) for mu in range(2)]
F, Fg = Ftx(A), Ftx(Ag)
print("[PASS] F'_{tx} = g^{-1} F_{tx} g:", simp(Fg - gi * F * g).is_zero())
print("[PASS] F itself changes (it is not gauge invariant):", not simp(Fg - F).is_zero())
print("[PASS] Tr(F_{tx} F_{tx}) is gauge invariant:", bool(((Fg * Fg).trace() - (F * F).trace()).simplify_full() == 0))

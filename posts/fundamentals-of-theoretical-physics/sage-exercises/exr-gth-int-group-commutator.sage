# Exercise exr-gth-int-group-commutator: for the constant SU(2) potential A_1 = c1 T_1, A_2 = c2 T_2, the group
# commutator of the transports along the sides of a small square, C = g1 g2 g1^{-1} g2^{-1} with g1 = exp(eps c1 T_1),
# g2 = exp(eps c2 T_2), is 1 + eps^2 c1 c2 T_3 + O(eps^3) = 1 + eps^2 F_{12} + O(eps^3); for an abelian potential C = 1.
eps, c1, c2 = var("eps c1 c2", domain="real")
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
T = [-I / 2 * sa for sa in s]
expT = lambda th, k: cos(th / 2) * identity_matrix(SR, 2) - I * sin(th / 2) * s[k]
g1, g1i, g2, g2i = expT(eps * c1, 0), expT(-eps * c1, 0), expT(eps * c2, 1), expT(-eps * c2, 1)
C = g1 * g2 * g1i * g2i
C2 = C.apply_map(lambda e: e.series(eps, 3).truncate())
F12 = c1 * T[0] * c2 * T[1] - c2 * T[1] * c1 * T[0]
print("C to second order in eps:"); print(C2.apply_map(lambda e: e.simplify_full()))
print("[PASS] F_12 = [A_1, A_2] = c1 c2 T_3:", (F12 - c1 * c2 * T[2]).is_zero())
print("[PASS] C = 1 + eps^2 F_12 + O(eps^3):", (C2 - identity_matrix(2) - eps^2 * F12).apply_map(lambda e: e.simplify_full()).is_zero())
h1, h2 = expT(eps * c1, 2), expT(eps * c2, 2)
print("[PASS] for two potentials along T_3 the transports commute, C = 1 exactly:", (h1 * h2 * expT(-eps * c1, 2) * expT(-eps * c2, 2)).apply_map(lambda e: e.simplify_full()) == identity_matrix(2))

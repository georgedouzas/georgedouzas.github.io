# Exercise exr-lg-su2-preimage: the two elements +-g of SU(2) covering the rotation by pi/2 about the X-axis: solve
# Ad_g = R_X(pi/2) for g = a0 1 + a1 (2T_1) + a2 (2T_2) + a3 (2T_3) and check that exactly the pair +-e^{(pi/2) T_1} remains.
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
T = [-I / 2 * sa for sa in s]
def Ad(g):
    gi = g.inverse()
    return matrix(SR, 3, 3, lambda a, b: (-2 * (T[a] * g * T[b] * gi).trace()).simplify_full())
R = matrix(SR, [[1, 0, 0], [0, 0, -1], [0, 1, 0]])          # rotation by pi/2 about X: e^{(pi/2) L_1}
a0, a1, a2, a3 = var("a0 a1 a2 a3", domain="real")
g = a0 * identity_matrix(SR, 2) + 2 * (a1 * T[0] + a2 * T[1] + a3 * T[2])
eqs = [e == 0 for e in (Ad(g) * (a0^2 + a1^2 + a2^2 + a3^2) - R * (a0^2 + a1^2 + a2^2 + a3^2)).list()] + [a0^2 + a1^2 + a2^2 + a3^2 == 1]
sols = solve(eqs, [a0, a1, a2, a3])
print("solutions (a0, a1, a2, a3):", sols)
g0 = (pi / 2 * T[0]).exp().apply_map(lambda e: e.simplify_full())
print("e^{(pi/2) T_1} =", g0.list(), "; Ad of it = R:", Ad(g0) == R, "; Ad of minus it = R:", Ad(-g0) == R)

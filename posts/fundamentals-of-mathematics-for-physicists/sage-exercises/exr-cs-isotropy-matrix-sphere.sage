# Exercise exr-cs-isotropy-matrix-sphere: the isotropy representation of SO(2) on the tangent plane of the sphere at
# the pole, computed twice: as the differential of h = exp(t L_3) on T_o S^2 = span(e_1, e_2), and as the adjoint
# action Ad_h on k = span(L_1, L_2); the identification X -> X . o intertwines them, h (X . o) = (Ad_h X) . o.
t = var("t", domain="real")
L = [matrix(SR, [[0,0,0],[0,0,-1],[0,1,0]]), matrix(SR, [[0,0,1],[0,0,0],[-1,0,0]]), matrix(SR, [[0,-1,0],[1,0,0],[0,0,0]])]
simp = lambda M: M.apply_map(lambda e: e.rectform().simplify_full())
h = simp((t * L[2]).exp()); o = vector(SR, [0, 0, 1])
dh = h.submatrix(0, 0, 2, 2)                      # h is linear, so dh_o = h restricted to the tangent plane z = 0
print("dh_o on T_o S^2 in the basis (e_1, e_2):", dh.rows())
coef = lambda X, b: (-(X * L[b]).trace() / 2).simplify_full()          # tr(L_a L_b) = -2 delta_ab
Ad = matrix(SR, 2, 2, lambda b, a: coef(simp(h * L[a] * h.inverse()), b))
print("Ad_h on k in the basis (L_1, L_2):", Ad.rows())
print("L_1 . o =", (L[0] * o).list(), " L_2 . o =", (L[1] * o).list())
print("h (X . o) = (Ad_h X) . o for X = L_1, L_2:", all(bool(simp(matrix(h * (L[a] * o)) - matrix(simp(h * L[a] * h.inverse()) * o)) == 0) for a in range(2)))
print("both matrices are rotations by the angle t (determinant 1):", bool(dh.det().simplify_full() == 1) and bool(Ad.det().simplify_full() == 1))

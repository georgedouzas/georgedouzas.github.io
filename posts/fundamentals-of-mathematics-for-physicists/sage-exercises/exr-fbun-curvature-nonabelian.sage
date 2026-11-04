# Exercise exr-fbun-curvature-nonabelian: the su(2) potential A_x = y T_1, A_y = x T_2 on the plane has curvature
# F_xy = d_x A_y - d_y A_x + [A_x, A_y] = T_2 - T_1 + x y T_3: derivative terms and a commutator term.
x, y = var("x y", domain="real")
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
T = [-I / 2 * sa for sa in s]
Ax, Ay = y * T[0], x * T[1]
F = diff(Ay, x) - diff(Ax, y) + Ax * Ay - Ay * Ax
def comps(M): return [(-2 * (M * T[a]).trace()).simplify_full() for a in range(3)]
print("components of F_xy in (T_1, T_2, T_3):", comps(F))
print("F_xy = T_2 - T_1 + x y T_3:", bool((F - (T[1] - T[0] + x * y * T[2])).simplify_full() == 0))
# gauge covariance at a point: with a constant g = e^{a T_1}, F' = g^{-1} F g and tr(F^2) is unchanged
a = var("a", domain="real")
g = (a * T[0]).exp().simplify_full()
Ax2, Ay2 = g.inverse() * Ax * g, g.inverse() * Ay * g
F2 = diff(Ay2, x) - diff(Ax2, y) + Ax2 * Ay2 - Ay2 * Ax2
print("F' = g^{-1} F g for constant g:", bool((F2 - g.inverse() * F * g).simplify_full() == 0))
print("tr(F_xy^2) =", (F * F).trace().simplify_full(), "; gauge invariant:", bool(((F2 * F2).trace() - (F * F).trace()).simplify_full() == 0))

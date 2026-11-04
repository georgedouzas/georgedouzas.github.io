# Exercise exr-eh-variation: the variation of sqrt(-g). For a diagonal metric g = diag(g0, g1, g2, g3), vary the inverse
# metric g^{mu mu} -> g^{mu mu} + eps d_mu and differentiate sqrt(-det g) at eps = 0; the result is
# -(1/2) sqrt(-g) g_{mu nu} delta g^{mu nu}. Then the same for a general symmetric metric via the trace formula.
eps = var("eps")
gd = list(var("g0 g1 g2 g3")); d = list(var("d0 d1 d2 d3"))
assume(g0 > 0); assume(g1 < 0); assume(g2 < 0); assume(g3 < 0)      # signature (+, -, -, -)
ginv = diagonal_matrix(SR, [1/gi + eps*di for gi, di in zip(gd, d)])
gvar = ginv.inverse()
sqrtg = sqrt(-gvar.det())
lhs = diff(sqrtg, eps).subs(eps=0).simplify_full()
rhs = (-1/2 * sqrt(-prod(gd)) * sum(gi*di for gi, di in zip(gd, d))).simplify_full()
print("d/d eps sqrt(-g) at eps = 0:", lhs)
print("[PASS] delta sqrt(-g) = -(1/2) sqrt(-g) g_{mu nu} delta g^{mu nu} for a diagonal metric:", bool((lhs - rhs).simplify_full() == 0))
# general 2x2 symmetric metric, to show the formula does not rely on diagonality
a, b, c = var("a b c"); da, db, dc = var("da db dc")
G = matrix(SR, [[a, b], [b, c]]); dGinv = matrix(SR, [[da, db], [db, dc]])
Ginv_var = G.inverse() + eps*dGinv
G_var = Ginv_var.inverse()
lhs2 = diff(sqrt(-G_var.det()), eps).subs(eps=0)
rhs2 = -1/2 * sqrt(-G.det()) * (G * dGinv).trace()     # g_{mu nu} delta g^{mu nu} = tr(g dg^{-1})
print("[PASS] the same identity for a general symmetric 2x2 metric:", bool((lhs2 - rhs2).simplify_full() == 0))

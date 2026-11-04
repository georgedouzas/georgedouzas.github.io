# Exercise exr-kk-inverse-metric: the inverse of the Kaluza-Klein ansatz metric. With G_{mu nu} = g_{mu nu} - phi^2 A_mu
# A_nu, G_{mu y} = -phi^2 A_mu, G_{yy} = -phi^2 (signature (+,-,-,-,-)), the inverse is G^{mu nu} = g^{mu nu},
# G^{mu y} = -A^mu, G^{yy} = -phi^{-2} + A_mu A^mu, and det G = -phi^2 det g. Checked on a generic symmetric 4D metric
# (symbolic entries) and on its determinant.
gs = {}
for i in range(4):
    for j in range(i, 4): gs[(i, j)] = gs[(j, i)] = var("g%d%d" % (i, j))
A = list(var("A0 A1 A2 A3")); phi = var("phi")
g = matrix(SR, 4, 4, lambda i, j: gs[(i, j)]); gi = g.inverse()
Aup = gi * vector(A)
G = block_matrix([[g - phi^2 * vector(A).column() * vector(A).row(), -phi^2 * vector(A).column()], [-phi^2 * vector(A).row(), matrix([[-phi^2]])]])
Ginv = block_matrix([[gi, -Aup.column()], [-Aup.row(), matrix([[-1 / phi^2 + vector(A) * Aup]])]])
print("[PASS] G^{-1} G = 1 for the block inverse of the ansatz metric (generic symmetric 4D metric):", (Ginv * G).simplify_rational() == identity_matrix(5))
print("[PASS] det G = -phi^2 det g:", bool((G.det() + phi^2 * g.det()).simplify_rational() == 0))

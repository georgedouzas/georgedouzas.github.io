# Exercise exr-sphere-metric-stereographic: the round metric in the stereographic chart, as the pullback of the
# Euclidean metric of R^3 through the inverse stereographic projection; conformally flat and regular at the pole.
E3 = Manifold(3, "R^3", structure="Riemannian"); cart = E3.chart("x y z")
delta = E3.metric("delta"); delta[0, 0] = 1; delta[1, 1] = 1; delta[2, 2] = 1
S = Manifold(2, "S", structure="Riemannian"); st = S.chart("u v"); u, v = st[:]
w = 1 + u**2 + v**2
Phi = S.diff_map(E3, {(st, cart): [2*u/w, 2*v/w, (u**2 + v**2 - 1)/w]})
g = S.metric("g"); g.set(Phi.pullback(delta))
Omega2 = 4/w**2
print("g_uu == 4/(1 + u^2 + v^2)^2:", bool((g[0, 0].expr() - Omega2).simplify_full() == 0), "; g_uv == 0:", bool(g[0, 1].expr().simplify_full() == 0), "; g_vv == g_uu:", bool((g[1, 1].expr() - g[0, 0].expr()).simplify_full() == 0))
print("determinant:", g.determinant().expr().factor())
print("inverse metric component g^uu:", g.inverse()[0, 0].expr().factor(), "; g^uv:", g.inverse()[0, 1].expr().simplify_full())
G0 = matrix([[g[i, j].expr().subs(u=0, v=0).simplify_full() for j in range(2)] for i in range(2)])
print("at u = v = 0 (the south pole (0, 0, -1), where the spherical chart fails) the matrix is", G0.list(), "with determinant", G0.det())

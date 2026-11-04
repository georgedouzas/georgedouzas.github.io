# Exercise exr-line-element-parabolic: the line element of the Euclidean plane in parabolic coordinates x = u v,
# y = (u^2 - v^2)/2, obtained with SageManifolds as the pullback of dx^2 + dy^2 through the coordinate map.
M = Manifold(2, "R^2", structure="Riemannian"); cart = M.chart("x y")
g = M.metric("g"); g[0, 0] = 1; g[1, 1] = 1
P = Manifold(2, "P", structure="Riemannian"); par = P.chart("u:(0,+oo) v:(0,+oo)"); u, v = par[:]
Phi = P.diff_map(M, {(par, cart): [u*v, (u**2 - v**2)/2]})
h = P.metric("h"); h.set(Phi.pullback(g))
print("line element in parabolic coordinates:", h.display())
print("components:"); print(h[:])
print("conformal to the flat chart metric with factor u^2 + v^2:", bool(h[0, 0].expr() == u**2 + v**2) and bool(h[1, 1].expr() == u**2 + v**2) and bool(h[0, 1].expr() == 0))
print("determinant:", h.determinant().expr().factor())
print("inverse metric:", h.inverse().display())

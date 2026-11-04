# Exercise exr-product-metric-inverse: the five-dimensional product metric eta - R^2 dphi^2 and a warped metric
# e^{2A(w)} eta - dw^2 declared as Lorentzian metrics in SageManifolds; determinants and block-diagonal inverses.
M5 = Manifold(5, "M5", structure="Lorentzian", signature="negative"); X = M5.chart(r"t x y z ph:(0,2*pi):\varphi"); t, x, y, z, ph = X[:]
R = var("R", domain="real"); assume(R > 0)
g = M5.metric("g"); g[0, 0] = 1; g[1, 1] = -1; g[2, 2] = -1; g[3, 3] = -1; g[4, 4] = -R**2
print("product metric:", g.display())
print("det g =", g.determinant().expr(), " (positive in five dimensions, four negative eigenvalues); sqrt|det g| =", sqrt(g.determinant().expr()).canonicalize_radical())
print("inverse:", g.inverse().display())
W = Manifold(5, "W", structure="Lorentzian", signature="negative"); Y = W.chart("t x y z w"); tw, xw, yw, zw, w = Y[:]
A = function("A")(w)
h = W.metric("h"); h[0, 0] = exp(2*A); h[1, 1] = -exp(2*A); h[2, 2] = -exp(2*A); h[3, 3] = -exp(2*A); h[4, 4] = -1
print("warped metric:", h.display())
print("det h =", h.determinant().expr().simplify_full())
print("inverse:", h.inverse().display())
print("the inverse is block diagonal with the reciprocal blocks:", all(h.inverse()[i, j].expr() == 0 for i in range(5) for j in range(5) if i != j))

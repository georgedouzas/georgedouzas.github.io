# Exercise exr-inverse-metric-check: inverting two constant metrics with SageManifolds, a Riemannian one with a cross
# term and a Lorentzian one with no diagonal time entry, and checking g^{ik} g_{kj} = delta^i_j.
M = Manifold(2, "M", structure="Riemannian"); X = M.chart("u v")
g = M.metric("g"); g[0, 0] = 4; g[0, 1] = 3; g[1, 1] = 9
print("g =", g.display()); print("det g =", g.determinant().expr()); print("inverse:", g.inverse().display())
print("g^{-1} g = 1:", bool(g.inverse()[:] * g[:] == identity_matrix(2)))
N = Manifold(2, "N", structure="Lorentzian", signature="negative"); Y = N.chart("t x")
h = N.metric("h"); h[0, 0] = 1; h[0, 1] = -1          # ds^2 = dt^2 - 2 dt dx, no dx^2 term
print("h =", h.display()); print("det h =", h.determinant().expr(), " (negative: Lorentzian)"); print("inverse:", h.inverse().display())
print("h^{-1} h = 1:", bool(h.inverse()[:] * h[:] == identity_matrix(2)))
print("eigenvalues of the matrix of h:", matrix(QQ, [[h[i, j].expr() for j in range(2)] for i in range(2)]).eigenvalues(), " -> one positive, one negative")

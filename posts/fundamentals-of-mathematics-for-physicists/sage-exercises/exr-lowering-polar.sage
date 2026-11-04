# Exercise exr-lowering-polar: lowering indices with the polar metric of the plane in SageManifolds (down): the
# rotation field d_phi becomes r^2 dphi, and X^flat(Y) = g(X, Y) for every Y.
M = Manifold(2, "R^2", structure="Riemannian"); U = M.open_subset("U")
pol = U.chart(r"r:(0,+oo) ph:(0,2*pi):\varphi"); r, ph = pol[:]
g = U.metric("g"); g[0, 0] = 1; g[1, 1] = r**2
dph = U.vector_field(0, 1, name="dph"); X = U.vector_field(1, 1, name="X"); Y = U.vector_field(r, -1, name="Y")
print("lowering d_phi:", dph.down(g).display())
print("lowering X = d_r + d_phi:", X.down(g).display(), "; components", X.down(g)[:])
print("X^flat(Y) =", X.down(g)(Y).expr(), "; g(X, Y) =", g(X, Y).expr(), "; equal:", bool(X.down(g)(Y).expr() == g(X, Y).expr()))
print("|d_phi| from the lowered form: sqrt((d_phi)^flat(d_phi)) =", sqrt(dph.down(g)(dph).expr()).simplify_full())

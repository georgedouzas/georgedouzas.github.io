# Exercise exr-wedge-rules: the graded rules of the wedge product and the exterior derivative, checked with
# SageManifolds on generic forms of the plane: alpha ^ beta = (-1)^{pq} beta ^ alpha and the Leibniz rule.
M = Manifold(2, "R^2"); X = M.chart("x y"); x, y = X[:]
a1, a2, b1, b2 = [function(n)(x, y) for n in ["a1", "a2", "b1", "b2"]]
al = M.one_form(a1, a2, name="alpha"); be = M.one_form(b1, b2, name="beta")
g = M.scalar_field(function("g")(x, y), name="g")
print("alpha ^ beta = -beta ^ alpha for one-forms:", al.wedge(be) == -be.wedge(al))
print("alpha ^ alpha = 0:", al.wedge(al) == 0)
lhs = (g * al).exterior_derivative(); rhs = g.differential().wedge(al) + g * al.exterior_derivative()
print("Leibniz rule d(g alpha) = dg ^ alpha + g d alpha:", lhs == rhs)
lhs2 = al.wedge(be).exterior_derivative()
print("d(alpha ^ beta) = d alpha ^ beta - alpha ^ d beta (both sides are 3-forms on a 2-manifold, hence zero):", lhs2 == 0)
print("d alpha =", al.exterior_derivative().display())

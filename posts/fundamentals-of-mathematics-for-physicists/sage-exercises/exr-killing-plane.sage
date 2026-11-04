# Exercise exr-killing-plane: the Killing vectors of the Euclidean plane and of the circle, checked with
# SageManifolds' lie_derivative: two translations and one rotation for the plane, one translation for the circle,
# and a field that is not Killing.
M = Manifold(2, "R^2", structure="Riemannian"); X = M.chart("x y"); x, y = X[:]
g = M.metric("g"); g[0, 0] = 1; g[1, 1] = 1
fields = {"d/dx": M.vector_field(1, 0), "d/dy": M.vector_field(0, 1), "x d/dy - y d/dx": M.vector_field(-y, x), "x d/dx (dilation)": M.vector_field(x, 0)}
for name, V in fields.items():
    print(f"L_({name}) g =", g.lie_derivative(V).display(), " -> Killing:", g.lie_derivative(V) == 0)
S1 = Manifold(1, "S^1", structure="Riemannian"); U = S1.open_subset("U"); c = U.chart("ph:(0,2*pi)"); ph = c[0]
h = U.metric("h"); h[0, 0] = 1
K = U.vector_field(function("k")(ph), name="K")
print("on the circle, L_K h =", h.lie_derivative(K).display(), " -> K is Killing iff k'(phi) = 0: K = const d/dphi")

# Exercise exr-maximal-hyperbolic: the hyperbolic plane (upper half-plane, g = (dx^2 + dy^2)/y^2) is maximally
# symmetric: d/dx, x d/dx + y d/dy and (x^2 - y^2) d/dx + 2xy d/dy are Killing, and their brackets close into sl(2, R);
# the Gauss curvature is the constant -1.
M = Manifold(2, "H2", structure="Riemannian"); C = M.chart("x y:(0,+oo)"); x, y = C[:]
g = M.metric("g"); g[0, 0] = 1 / y ** 2; g[1, 1] = 1 / y ** 2
P = M.vector_field(1, 0, name="P"); D = M.vector_field(x, y, name="D"); S = M.vector_field(x ** 2 - y ** 2, 2 * x * y, name="S")
for name, V in [("P", P), ("D", D), ("S", S)]:
    print(f"L_{name} g =", g.lie_derivative(V).display(), " -> Killing:", g.lie_derivative(V) == 0)
print("[P, D] = P:", P.bracket(D) == P, ";  [P, S] = 2 D:", P.bracket(S) == 2 * D, ";  [D, S] = S:", D.bracket(S) == S)
print("scalar curvature R =", g.ricci_scalar().expr(), " (Gauss curvature R/2 = -1)")
print("the Euclidean rotation -y d/dx + x d/dy is Killing here:", g.lie_derivative(M.vector_field(-y, x)) == 0)

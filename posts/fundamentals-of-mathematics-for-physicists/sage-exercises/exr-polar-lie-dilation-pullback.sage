# Exercise exr-polar-lie-dilation-pullback: the Lie derivative as the derivative of the pullback along the flow,
# L_X g = d/dt|_{t=0} Phi_t^* g, checked for the rotation and the dilation of the plane in Cartesian coordinates,
# where Phi_t^* g = g and Phi_t^* g = e^{2t} g respectively.
t = var("t", domain="real")
M = Manifold(2, "R^2", structure="Riemannian"); C = M.chart("x y"); x, y = C[:]
g = M.metric("g"); g[0, 0] = 1; g[1, 1] = 1
fields = {"rotation": (M.vector_field(-y, x), [x * cos(t) - y * sin(t), x * sin(t) + y * cos(t)]),
          "dilation": (M.vector_field(x, y), [exp(t) * x, exp(t) * y])}
for name, (V, comps) in fields.items():
    Phi = M.diff_map(M, comps, name="Phi")
    pb = Phi.pullback(g)
    pb.apply_map(lambda e: e.simplify_full())
    deriv = [[diff(pb[i, j].expr(), t).subs(t=0) for j in range(2)] for i in range(2)]
    L = g.lie_derivative(V)
    print(f"{name}: Phi_t^* g =", pb.display())
    print(f"   d/dt|0 Phi_t^* g = {deriv};  L_X g = {[[L[i, j].expr() for j in range(2)] for i in range(2)]};  equal:",
          all(bool(deriv[i][j] == L[i, j].expr()) for i in range(2) for j in range(2)))

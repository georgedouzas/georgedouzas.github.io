# Exercise exr-isometry-algebra-plane: the Lie algebra of the Euclidean group E(2), spanned by the Killing fields
# P1 = d/dx, P2 = d/dy, J = x d/dy - y d/dx of the plane; the brackets close, and the dilation D = x d/dx + y d/dy
# (not Killing) satisfies [D, P_i] = -P_i, [D, J] = 0.
M = Manifold(2, "R^2", structure="Riemannian"); C = M.chart("x y"); x, y = C[:]
g = M.metric("g"); g[0, 0] = 1; g[1, 1] = 1
F = {"P1": M.vector_field(1, 0), "P2": M.vector_field(0, 1), "J": M.vector_field(-y, x)}
for a, b in [("P1", "P2"), ("J", "P1"), ("J", "P2")]:
    br = F[a].bracket(F[b])
    name = [n for n, V in F.items() if br == V] + ["-" + n for n, V in F.items() if br == -V] + (["0"] if br == 0 else [])
    print(f"[{a}, {b}] =", br.display(), " =", name[0], ";  Killing:", g.lie_derivative(br) == 0)
D = M.vector_field(x, y)
print("[D, P1] = -P1:", D.bracket(F["P1"]) == -F["P1"], ";  [D, P2] = -P2:", D.bracket(F["P2"]) == -F["P2"], ";  [D, J] = 0:", D.bracket(F["J"]) == 0)
print("L_D g = 2 g (conformal, not Killing):", g.lie_derivative(D) == 2 * g)

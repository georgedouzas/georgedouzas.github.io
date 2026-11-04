# Exercise exr-kkr-isometry-killing: d_y is a Killing vector of the ansatz, eps(x) d_y is not unless eps is constant,
# and the Lie derivative along eps(x) d_y is the variation of the ansatz under delta A = d eps. Checked with
# SageManifolds in 2 + 1 dimensions (two spacetime coordinates t, x and the circle y), signature (+,-,-).
M = Manifold(3, "M", structure="Lorentzian"); X = M.chart("t x y"); t, x, y = X[:]
gtt, gtx, gxx, phi, A0, A1, eps = [function(n)(t, x) for n in ("gtt", "gtx", "gxx", "phi", "A0", "A1", "eps")]
gs = [[gtt, gtx], [gtx, gxx]]; A = [A0, A1]
def ansatz(name, A):
    G = M.metric(name)
    for i in range(2):
        for j in range(i, 2): G[i, j] = gs[i][j] - phi^2 * A[i] * A[j]
        G[i, 2] = -phi^2 * A[i]
    G[2, 2] = -phi^2
    return G
G = ansatz("G", A)
dy = M.vector_field(0, 0, 1)
print("[PASS] the Lie derivative of the ansatz along d_y vanishes (d_y is Killing):", all(bool(G.lie_derivative(dy)[i, j].expr() == 0) for i in range(3) for j in range(3)))
xi = M.vector_field(0, 0, eps)
L = G.lie_derivative(xi)
dA = [diff(eps, t), diff(eps, x)]
var("s")
Gs = ansatz("Gs", [A[i] + s * dA[i] for i in range(2)])
ok = all(bool((L[i, j].expr() - diff(Gs[i, j].expr(), s).subs(s=0)).simplify_full() == 0) for i in range(3) for j in range(3))
print("L_xi G_{t y} =", L[0, 2].expr())
print("[PASS] L_{eps d_y} G = d/ds G(A + s d eps) at s = 0: the Lie derivative is the gauge variation delta A = d eps:", ok)

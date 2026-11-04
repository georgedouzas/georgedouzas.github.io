# Exercise exr-kkr-weyl-de-sitter: de Sitter space as a Weyl rescaling of Minkowski space. In conformal time tau < 0,
# g~ = Omega^2 eta with Omega = -1/(H tau); signature (+,-,-,-). SageManifolds gives R~ = -12 H^2, and the
# four-dimensional formula R~ = Omega^{-2} R - 6 Omega^{-3} Box Omega with R = 0 and Box = d_tau^2 - nabla^2 agrees.
M = Manifold(4, "M", structure="Lorentzian"); X = M.chart("tau x y z"); tau, x, y, z = X[:]
H = var("H"); assume(H > 0); assume(tau < 0)
Om = -1 / (H * tau)
g = M.metric("g"); g[0, 0] = Om^2
for i in (1, 2, 3): g[i, i] = -Om^2
R = g.ricci_scalar().expr().simplify_full()
print("R~ =", R)
print("[PASS] R~ = -12 H^2 = -6 Omega^{-3} d_tau^2 Omega:", bool(R == -12 * H^2) and bool((-6 * diff(Om, tau, 2) / Om^3 + 12 * H^2).simplify_full() == 0))

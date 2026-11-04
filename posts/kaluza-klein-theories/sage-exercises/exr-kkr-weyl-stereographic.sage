# Exercise exr-kkr-weyl-stereographic: the round sphere is conformally flat. The flat plane rescaled by
# Omega = 2/(1 + x^2 + y^2) (the stereographic metric) has curvature R = 2, and the two-dimensional Weyl formula
# R~ = Omega^{-2}(R - 2 Delta ln Omega) with R = 0 and the flat Laplacian Delta reproduces it.
M = Manifold(2, "M"); X = M.chart("x y"); x, y = X[:]
Om = 2 / (1 + x^2 + y^2)
g = M.metric("g"); g[0, 0] = Om^2; g[1, 1] = Om^2
R = g.ricci_scalar().expr().simplify_full()
lap = (diff(log(Om), x, 2) + diff(log(Om), y, 2)).simplify_full()
print("R of the stereographic metric:", R, "; Delta ln Omega =", lap)
print("[PASS] R = 2 = Omega^{-2}(0 - 2 Delta ln Omega):", bool(R == 2) and bool((-2 * lap / Om^2 - 2).simplify_full() == 0))

# Exercise exr-kkr-weyl-three-dims: the n-dimensional Weyl formula R~ = Omega^{-2}(R - 2(n-1) Delta ln Omega -
# (n-1)(n-2)|d ln Omega|^2) for n = 3, on flat Euclidean space with Omega = Omega(x): the coefficients are 4 and 2.
M = Manifold(3, "M"); X = M.chart("x y z"); x, y, z = X[:]
Om = function("Omega")(x)
g = M.metric("g")
for i in range(3): g[i, i] = Om^2
R = g.ricci_scalar().expr()
L = log(Om)
target = (-4 * diff(L, x, 2) - 2 * diff(L, x)^2) / Om^2
print("R~ =", R.simplify_full())
print("[PASS] n = 3: R~ = Omega^{-2}(-4 (ln Omega)'' - 2 ((ln Omega)')^2) on flat space:", bool((R - target).simplify_full() == 0))

# Exercise exr-kkr-shift-remove-a: in two dimensions the gauge field can be removed by a reparametrisation of the
# circle. For the Riemannian toy dx^2 + phi(x)^2 (dy + A(x) dx)^2 the shift y' = y + eps(x) with eps' = A, i.e.
# eps = int A, gives the metric dx^2 + phi^2 dy'^2: A' = A - eps' = 0.
M = Manifold(2, "M"); X = M.chart("x y"); x, y = X[:]
Xp = M.chart("xp yp"); xp, yp = Xp[:]
phi = function("phi"); a = function("a")                    # a = antiderivative of A, A = a'
X_to_Xp = X.transition_map(Xp, [x, y + a(x)]); X_to_Xp.set_inverse(xp, yp - a(xp))
A = diff(a(x), x)
g = M.metric("g"); g[0, 0] = 1 + phi(x)^2 * A^2; g[0, 1] = phi(x)^2 * A; g[1, 1] = phi(x)^2
c = g.comp(Xp.frame())
print("components in (x, y'):", c[0, 0, Xp].expr().simplify_full(), ",", c[0, 1, Xp].expr().simplify_full(), ",", c[1, 1, Xp].expr().simplify_full())
print("[PASS] with eps = int A the metric becomes dx^2 + phi^2 dy'^2:", bool(c[0, 0, Xp].expr().simplify_full() == 1) and bool(c[0, 1, Xp].expr().simplify_full() == 0) and bool((c[1, 1, Xp].expr() - phi(xp)^2).simplify_full() == 0))

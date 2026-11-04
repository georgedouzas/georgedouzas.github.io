# Exercise exr-emm-lorenz-residual: plane-wave gauge functions with box lambda = 0.
t, x, y, z = var("t x y z")
X = [x, y, z]
def div(F): return sum(diff(F[i], X[i]) for i in range(3)).simplify_full()
def curl(F): return vector([diff(F[2], y) - diff(F[1], z), diff(F[0], z) - diff(F[2], x), diff(F[1], x) - diff(F[0], y)]).simplify_full()
def grad(f): return vector([diff(f, v) for v in X])
box = lambda f: diff(f, t, 2) - sum(diff(f, v, 2) for v in X)
lap = lambda f: sum(diff(f, v, 2) for v in X)
k1, k2, k3, w = var('k1 k2 k3 w')
lam = sin(k1 * x + k2 * y + k3 * z - w * t)
print('box lambda / lambda =', (box(lam) / lam).simplify_full())
print('solutions for w:', solve(box(lam) / lam == 0, w))

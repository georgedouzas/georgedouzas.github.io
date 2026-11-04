# Exercise exr-emm-lorenz-derive: -lap phi - d_t div A = box phi - d_t (d_t phi + div A).
t, x, y, z = var("t x y z")
X = [x, y, z]
def div(F): return sum(diff(F[i], X[i]) for i in range(3)).simplify_full()
def curl(F): return vector([diff(F[2], y) - diff(F[1], z), diff(F[0], z) - diff(F[2], x), diff(F[1], x) - diff(F[0], y)]).simplify_full()
def grad(f): return vector([diff(f, v) for v in X])
box = lambda f: diff(f, t, 2) - sum(diff(f, v, 2) for v in X)
lap = lambda f: sum(diff(f, v, 2) for v in X)
phi = function('phi')(t, x, y, z); A = vector([function('A%d' % i)(t, x, y, z) for i in range(3)])
print('difference:', (-lap(phi) - diff(div(A), t) - (box(phi) - diff(diff(phi, t) + div(A), t))).simplify_full())

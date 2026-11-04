# Exercise exr-emm-gauge-check: E and B are unchanged by a gauge transformation.
t, x, y, z = var("t x y z")
X = [x, y, z]
def div(F): return sum(diff(F[i], X[i]) for i in range(3)).simplify_full()
def curl(F): return vector([diff(F[2], y) - diff(F[1], z), diff(F[0], z) - diff(F[2], x), diff(F[1], x) - diff(F[0], y)]).simplify_full()
def grad(f): return vector([diff(f, v) for v in X])
box = lambda f: diff(f, t, 2) - sum(diff(f, v, 2) for v in X)
lap = lambda f: sum(diff(f, v, 2) for v in X)
phi = function('phi')(t, x, y, z); A = vector([function('A%d' % i)(t, x, y, z) for i in range(3)]); lam = function('lam')(t, x, y, z)
E = lambda ph, AA: -grad(ph) - diff(AA, t)
phi2 = phi - diff(lam, t); A2 = A + grad(lam)
print('change of E:', (E(phi2, A2) - E(phi, A)).simplify_full(), '; change of B:', (curl(A2) - curl(A)).simplify_full())

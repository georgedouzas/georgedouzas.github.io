# Exercise exr-emm-gauge-landau: gauge functions relating three potentials of a uniform field B e_z.
t, x, y, z = var("t x y z")
X = [x, y, z]
def div(F): return sum(diff(F[i], X[i]) for i in range(3)).simplify_full()
def curl(F): return vector([diff(F[2], y) - diff(F[1], z), diff(F[0], z) - diff(F[2], x), diff(F[1], x) - diff(F[0], y)]).simplify_full()
def grad(f): return vector([diff(f, v) for v in X])
box = lambda f: diff(f, t, 2) - sum(diff(f, v, 2) for v in X)
lap = lambda f: sum(diff(f, v, 2) for v in X)
B = var('B')
As = B / 2 * vector([-y, x, 0]); A1 = vector([0, B * x, 0]); A2 = vector([-B * y, 0, 0])
for lam, src, dst, name in [(B*x*y/2, As, A1, 'symmetric -> (0, Bx, 0)'), (-B*x*y/2, As, A2, 'symmetric -> (-By, 0, 0)'), (-B*x*y, A1, A2, '(0, Bx, 0) -> (-By, 0, 0)')]:
    print(name, ': lambda =', lam, ', check:', (src + grad(lam) - dst).simplify_full())

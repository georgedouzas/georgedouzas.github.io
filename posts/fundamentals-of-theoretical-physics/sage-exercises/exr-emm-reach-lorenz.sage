# Exercise exr-emm-reach-lorenz: reaching the Lorenz gauge from phi = 0, A = (t x, 0, 0).
t, x, y, z = var("t x y z")
X = [x, y, z]
def div(F): return sum(diff(F[i], X[i]) for i in range(3)).simplify_full()
def curl(F): return vector([diff(F[2], y) - diff(F[1], z), diff(F[0], z) - diff(F[2], x), diff(F[1], x) - diff(F[0], y)]).simplify_full()
def grad(f): return vector([diff(f, v) for v in X])
box = lambda f: diff(f, t, 2) - sum(diff(f, v, 2) for v in X)
lap = lambda f: sum(diff(f, v, 2) for v in X)
phi = 0; A = vector([t * x, 0, 0])
lam = t^3 / 6
print('d_t phi + div A =', diff(phi, t) + div(A), '; box lambda =', box(lam))
phi2 = phi - diff(lam, t); A2 = A + grad(lam)
print('phi_new =', phi2, '; A_new =', A2, '; Lorenz condition:', diff(phi2, t) + div(A2))

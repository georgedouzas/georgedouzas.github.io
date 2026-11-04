# Exercise exr-emm-reach-coulomb: reaching the Coulomb gauge from phi = 0, A = (0, x^2, z t).
t, x, y, z = var("t x y z")
X = [x, y, z]
def div(F): return sum(diff(F[i], X[i]) for i in range(3)).simplify_full()
def curl(F): return vector([diff(F[2], y) - diff(F[1], z), diff(F[0], z) - diff(F[2], x), diff(F[1], x) - diff(F[0], y)]).simplify_full()
def grad(f): return vector([diff(f, v) for v in X])
box = lambda f: diff(f, t, 2) - sum(diff(f, v, 2) for v in X)
lap = lambda f: sum(diff(f, v, 2) for v in X)
phi = 0; A = vector([0, x^2, z * t])
lam = -t * (x^2 + y^2 + z^2) / 6
print('div A =', div(A), '; lap lambda =', lap(lam))
phi2 = phi - diff(lam, t); A2 = A + grad(lam)
print('phi_new =', phi2, '; A_new =', A2, '; div A_new =', div(A2))
print('fields unchanged:', (-grad(phi2) - diff(A2, t) + diff(A, t)).simplify_full(), (curl(A2) - curl(A)).simplify_full())

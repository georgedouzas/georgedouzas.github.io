# Exercise exr-emm-lorenz-continuity: with rho = box phi, J = box A, continuity is box of the Lorenz expression.
t, x, y, z = var("t x y z")
X = [x, y, z]
def div(F): return sum(diff(F[i], X[i]) for i in range(3)).simplify_full()
def curl(F): return vector([diff(F[2], y) - diff(F[1], z), diff(F[0], z) - diff(F[2], x), diff(F[1], x) - diff(F[0], y)]).simplify_full()
def grad(f): return vector([diff(f, v) for v in X])
box = lambda f: diff(f, t, 2) - sum(diff(f, v, 2) for v in X)
lap = lambda f: sum(diff(f, v, 2) for v in X)
phi = function('phi')(t, x, y, z); A = vector([function('A%d' % i)(t, x, y, z) for i in range(3)])
rho = box(phi); J = vector([box(Ai) for Ai in A])
print('d_t rho + div J - box(d_t phi + div A) =', (diff(rho, t) + div(J) - box(diff(phi, t) + div(A))).simplify_full())

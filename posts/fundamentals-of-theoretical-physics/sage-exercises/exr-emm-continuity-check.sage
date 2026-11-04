# Exercise exr-emm-continuity-check: a charge cloud moving rigidly with velocity v satisfies continuity.
t, x, y, z = var("t x y z")
X = [x, y, z]
def div(F): return sum(diff(F[i], X[i]) for i in range(3)).simplify_full()
def curl(F): return vector([diff(F[2], y) - diff(F[1], z), diff(F[0], z) - diff(F[2], x), diff(F[1], x) - diff(F[0], y)]).simplify_full()
def grad(f): return vector([diff(f, v) for v in X])
v1, v2, v3 = var('v1 v2 v3')
f = function('f')
rho = f(x - v1 * t, y - v2 * t, z - v3 * t)
J = vector([v1, v2, v3]) * rho
print('d rho/dt + div J =', (diff(rho, t) + div(J)).simplify_full())

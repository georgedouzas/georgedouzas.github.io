# Exercise exr-emm-find-sources: the sources of E = (x t, 0, 0), B = (0, 0, x^2 y).
t, x, y, z = var("t x y z")
X = [x, y, z]
def div(F): return sum(diff(F[i], X[i]) for i in range(3)).simplify_full()
def curl(F): return vector([diff(F[2], y) - diff(F[1], z), diff(F[0], z) - diff(F[2], x), diff(F[1], x) - diff(F[0], y)]).simplify_full()
def grad(f): return vector([diff(f, v) for v in X])
E = vector([x * t, 0, 0]); B = vector([0, 0, x^2 * y])
rho = div(E); J = (curl(B) - diff(E, t)).simplify_full()
print('rho =', rho, '; J =', J)
print('div B =', div(B), '; curl E + dB/dt =', curl(E) + diff(B, t), '; continuity:', diff(rho, t) + div(J))

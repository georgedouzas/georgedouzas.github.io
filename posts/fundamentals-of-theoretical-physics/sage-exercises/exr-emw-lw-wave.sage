# Exercise exr-emw-lw-wave: the potential of a uniformly moving charge solves the wave equation and the Lorenz condition.
t, x, y, z = var("t x y z")
X = [x, y, z]
def div(F): return sum(diff(F[i], X[i]) for i in range(3)).simplify_full()
def curl(F): return vector([diff(F[2], y) - diff(F[1], z), diff(F[0], z) - diff(F[2], x), diff(F[1], x) - diff(F[0], y)]).simplify_full()
def grad(f): return vector([diff(f, v) for v in X])
def maxwell(E, B): return [div(E), div(B), (curl(E) + diff(B, t)).simplify_full(), (curl(B) - diff(E, t)).simplify_full()]
box = lambda f: (diff(f, t, 2) - sum(diff(f, v, 2) for v in X)).simplify_full()
q, v = var('q v'); assume(v > 0, v < 1)
phi = q / (4 * pi * sqrt((x - v * t)^2 + (1 - v^2) * (y^2 + z^2)))
A = vector([v * phi, 0, 0])
print('box phi =', box(phi), '; Lorenz condition d_t phi + div A =', (diff(phi, t) + div(A)).simplify_full())

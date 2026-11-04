# Exercise exr-emm-gauge-pure-wave: a pure gauge potential wave produces no fields.
t, x, y, z = var("t x y z")
X = [x, y, z]
def div(F): return sum(diff(F[i], X[i]) for i in range(3)).simplify_full()
def curl(F): return vector([diff(F[2], y) - diff(F[1], z), diff(F[0], z) - diff(F[2], x), diff(F[1], x) - diff(F[0], y)]).simplify_full()
def grad(f): return vector([diff(f, v) for v in X])
box = lambda f: diff(f, t, 2) - sum(diff(f, v, 2) for v in X)
lap = lambda f: sum(diff(f, v, 2) for v in X)
f = function('f'); E0 = var('E0')
phi = diff(f(z - t), z); A = vector([0, 0, diff(f(z - t), z)])
print('pure gauge part: E =', (-grad(phi) - diff(A, t)).simplify_full(), ', B =', curl(A))
A2 = A + vector([E0 * sin(z - t), 0, 0])
print('with the plane wave added: E =', (-grad(phi) - diff(A2, t)).simplify_full(), ', B =', curl(A2))

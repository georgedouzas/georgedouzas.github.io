# Exercise exr-emw-lw-field-uniform: the fields of a uniformly moving charge from its potentials.
t, x, y, z = var("t x y z")
X = [x, y, z]
def div(F): return sum(diff(F[i], X[i]) for i in range(3)).simplify_full()
def curl(F): return vector([diff(F[2], y) - diff(F[1], z), diff(F[0], z) - diff(F[2], x), diff(F[1], x) - diff(F[0], y)]).simplify_full()
def grad(f): return vector([diff(f, v) for v in X])
def maxwell(E, B): return [div(E), div(B), (curl(E) + diff(B, t)).simplify_full(), (curl(B) - diff(E, t)).simplify_full()]
box = lambda f: (diff(f, t, 2) - sum(diff(f, v, 2) for v in X)).simplify_full()
q, v = var('q v'); assume(v > 0, v < 1)
g = 1 / sqrt(1 - v^2)
phi = q * g / (4 * pi * sqrt(g^2 * (x - v * t)^2 + y^2 + z^2)); A = vector([v * phi, 0, 0])
E = (-grad(phi) - diff(A, t)).simplify_full(); B = curl(A)
Eexp = q * g * vector([x - v * t, y, z]) / (4 * pi * (g^2 * (x - v * t)^2 + y^2 + z^2)^(3/2))
print('E - expected =', (E - Eexp).simplify_full(), '; B - v x E =', (B - vector([v, 0, 0]).cross_product(E)).simplify_full())

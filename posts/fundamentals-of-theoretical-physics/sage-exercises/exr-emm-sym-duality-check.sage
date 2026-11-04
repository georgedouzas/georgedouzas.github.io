# Exercise exr-emm-sym-duality-check: duality E -> B, B -> -E maps the vacuum equations to themselves.
t, x, y, z = var("t x y z")
X = [x, y, z]
def div(F): return sum(diff(F[i], X[i]) for i in range(3)).simplify_full()
def curl(F): return vector([diff(F[2], y) - diff(F[1], z), diff(F[0], z) - diff(F[2], x), diff(F[1], x) - diff(F[0], y)]).simplify_full()
def grad(f): return vector([diff(f, v) for v in X])
def maxwell(E, B): return [div(E), div(B), (curl(E) + diff(B, t)).simplify_full(), (curl(B) - diff(E, t)).simplify_full()]
E = vector([function('E%d' % i)(t, x, y, z) for i in range(3)]); B = vector([function('B%d' % i)(t, x, y, z) for i in range(3)])
m = maxwell(E, B); md = maxwell(B, -E)
print('div E\' - div B =', (md[0] - m[1]).simplify_full(), '; div B\' + div E =', (md[1] + m[0]).simplify_full())
print('Faraday\' - Ampere =', (md[2] - m[3]).simplify_full(), '; Ampere\' + Faraday =', (md[3] + m[2]).simplify_full())

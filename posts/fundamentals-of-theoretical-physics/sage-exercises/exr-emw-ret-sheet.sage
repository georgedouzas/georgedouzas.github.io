# Exercise exr-emw-ret-sheet: fields of a current sheet switched on at t = 0, behind the front (t > z > 0).
t, x, y, z = var("t x y z")
X = [x, y, z]
def div(F): return sum(diff(F[i], X[i]) for i in range(3)).simplify_full()
def curl(F): return vector([diff(F[2], y) - diff(F[1], z), diff(F[0], z) - diff(F[2], x), diff(F[1], x) - diff(F[0], y)]).simplify_full()
def grad(f): return vector([diff(f, v) for v in X])
def maxwell(E, B): return [div(E), div(B), (curl(E) + diff(B, t)).simplify_full(), (curl(B) - diff(E, t)).simplify_full()]
box = lambda f: (diff(f, t, 2) - sum(diff(f, v, 2) for v in X)).simplify_full()
K0 = var('K0', domain='positive')
A = vector([K0 / 2 * (t - z), 0, 0])
E = -diff(A, t); B = curl(A)
print('E =', E, '; B =', B, '; vacuum equations:', maxwell(E, B))
S = E.cross_product(B)
print('Poynting vector:', S, '; work done by the source per area and time, -E.K =', -E[0] * K0, '= flux out of both sides 2 S_z =', 2 * S[2])

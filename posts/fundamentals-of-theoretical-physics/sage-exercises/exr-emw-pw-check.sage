# Exercise exr-emw-pw-check: conditions on E0 cos(k.x - w t), B0 cos(k.x - w t) from the vacuum Maxwell equations.
t, x, y, z = var("t x y z")
X = [x, y, z]
def div(F): return sum(diff(F[i], X[i]) for i in range(3)).simplify_full()
def curl(F): return vector([diff(F[2], y) - diff(F[1], z), diff(F[0], z) - diff(F[2], x), diff(F[1], x) - diff(F[0], y)]).simplify_full()
def grad(f): return vector([diff(f, v) for v in X])
def maxwell(E, B): return [div(E), div(B), (curl(E) + diff(B, t)).simplify_full(), (curl(B) - diff(E, t)).simplify_full()]
k = vector(var('k1 k2 k3')); E0 = vector(var('e1 e2 e3')); B0 = vector(var('b1 b2 b3')); w = var('w')
ph = k.dot_product(vector(X)) - w * t
E = E0 * cos(ph); B = B0 * cos(ph)
m = maxwell(E, B)
print('div E / (-sin) =', (m[0] / (-sin(ph))).simplify_full(), '; div B / (-sin) =', (m[1] / (-sin(ph))).simplify_full())
print('Faraday / (-sin) =', (m[2] / (-sin(ph))).simplify_full(), '= k x E0 - w B0')
print('Ampere / (-sin) =', (m[3] / (-sin(ph))).simplify_full(), '= k x B0 + w E0')
# with B0 = k x E0/w and k.E0 = 0, Ampere gives (k.E0) k - |k|^2 E0 + w^2 E0 = (w^2 - |k|^2) E0
Bsub = k.cross_product(E0) / w
print('k x (k x E0)/w + w E0 =', (k.cross_product(Bsub) + w * E0).simplify_full())

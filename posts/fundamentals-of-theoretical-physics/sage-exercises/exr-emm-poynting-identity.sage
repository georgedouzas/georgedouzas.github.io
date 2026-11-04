# Exercise exr-emm-poynting-identity: div(E x B) = B . curl E - E . curl B for arbitrary fields.
t, x, y, z = var("t x y z")
X = [x, y, z]
def div(F): return sum(diff(F[i], X[i]) for i in range(3)).simplify_full()
def curl(F): return vector([diff(F[2], y) - diff(F[1], z), diff(F[0], z) - diff(F[2], x), diff(F[1], x) - diff(F[0], y)]).simplify_full()
def grad(f): return vector([diff(f, v) for v in X])
E = vector([function('E%d' % i)(t, x, y, z) for i in range(3)]); B = vector([function('B%d' % i)(t, x, y, z) for i in range(3)])
print('difference:', (div(E.cross_product(B)) - B.dot_product(curl(E)) + E.dot_product(curl(B))).simplify_full())

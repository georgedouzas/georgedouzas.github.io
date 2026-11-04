# Exercise exr-emm-stress-identity: (E x curl E)_i = d_j (E^2 delta_ij/2 - E_i E_j) + E_i div E.
t, x, y, z = var("t x y z")
X = [x, y, z]
def div(F): return sum(diff(F[i], X[i]) for i in range(3)).simplify_full()
def curl(F): return vector([diff(F[2], y) - diff(F[1], z), diff(F[0], z) - diff(F[2], x), diff(F[1], x) - diff(F[0], y)]).simplify_full()
def grad(f): return vector([diff(f, v) for v in X])
E = vector([function('E%d' % i)(x, y, z) for i in range(3)])
E2 = E.dot_product(E)
lhs = E.cross_product(curl(E))
rhs = vector([sum(diff(E2 / 2 * (1 if i == j else 0) - E[i] * E[j], X[j]) for j in range(3)) + E[i] * div(E) for i in range(3)])
print('difference:', (lhs - rhs).simplify_full())

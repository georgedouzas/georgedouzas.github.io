# Exercise exr-emm-pot-derive: Maxwell's equations in terms of arbitrary potentials.
t, x, y, z = var("t x y z")
X = [x, y, z]
def div(F): return sum(diff(F[i], X[i]) for i in range(3)).simplify_full()
def curl(F): return vector([diff(F[2], y) - diff(F[1], z), diff(F[0], z) - diff(F[2], x), diff(F[1], x) - diff(F[0], y)]).simplify_full()
def grad(f): return vector([diff(f, v) for v in X])
phi = function('phi')(t, x, y, z); A = vector([function('A%d' % i)(t, x, y, z) for i in range(3)])
E = -grad(phi) - diff(A, t); B = curl(A)
box = lambda f: diff(f, t, 2) - sum(diff(f, v, 2) for v in X)
lap = lambda f: sum(diff(f, v, 2) for v in X)
print('div E - (-lap phi - d_t div A) =', (div(E) - (-lap(phi) - diff(div(A), t))).simplify_full())
lhs2 = vector([box(A[i]) for i in range(3)]) + grad(div(A) + diff(phi, t))
print('curl B - dE/dt - (box A + grad(div A + d_t phi)) =', (curl(B) - diff(E, t) - lhs2).simplify_full())
print('homogeneous equations: div B =', div(B), '; curl E + dB/dt =', (curl(E) + diff(B, t)).simplify_full())

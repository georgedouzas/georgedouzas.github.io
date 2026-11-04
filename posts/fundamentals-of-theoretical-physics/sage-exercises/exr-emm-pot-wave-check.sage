# Exercise exr-emm-pot-wave-check: fields of the potential A = E0 sin(z - t) e_x, phi = 0.
t, x, y, z = var("t x y z")
X = [x, y, z]
def div(F): return sum(diff(F[i], X[i]) for i in range(3)).simplify_full()
def curl(F): return vector([diff(F[2], y) - diff(F[1], z), diff(F[0], z) - diff(F[2], x), diff(F[1], x) - diff(F[0], y)]).simplify_full()
def grad(f): return vector([diff(f, v) for v in X])
E0 = var('E0')
phi = 0; A = vector([E0 * sin(z - t), 0, 0])
E = -grad(phi) - diff(A, t); B = curl(A)
print('E =', E, '; B =', B)
print('vacuum Maxwell:', div(E), div(B), (curl(E) + diff(B, t)).simplify_full(), (curl(B) - diff(E, t)).simplify_full())
box = lambda f: diff(f, t, 2) - sum(diff(f, v, 2) for v in X)
print('potential equations: -lap phi - d_t div A =', -sum(diff(phi, v, 2) for v in X) - diff(div(A), t), '; box A + grad(div A + d_t phi) =', (vector([box(Ai) for Ai in A]) + grad(div(A) + diff(phi, t))).simplify_full())

# Exercise exr-emm-pot-two-growing: two vector potentials of B(t) e_z, both with phi = 0.
t, x, y, z = var("t x y z")
X = [x, y, z]
def div(F): return sum(diff(F[i], X[i]) for i in range(3)).simplify_full()
def curl(F): return vector([diff(F[2], y) - diff(F[1], z), diff(F[0], z) - diff(F[2], x), diff(F[1], x) - diff(F[0], y)]).simplify_full()
def grad(f): return vector([diff(f, v) for v in X])
B = function('B')(t)
A1 = B / 2 * vector([-y, x, 0]); A2 = vector([0, B * x, 0])
print('curl A1 =', curl(A1), '; curl A2 =', curl(A2))
E1 = -diff(A1, t); E2 = -diff(A2, t)
print('E1 =', E1, '; E2 =', E2)
print('E2 - E1 + (dB/dt/2) grad(x y) =', (E2 - E1 + diff(B, t) / 2 * grad(x * y)).simplify_full())
phi2 = -diff(B, t) * x * y / 2
print('with phi2 = -dB/dt x y/2: E2 - grad(phi2) - E1 =', (E2 - grad(phi2) - E1).simplify_full())

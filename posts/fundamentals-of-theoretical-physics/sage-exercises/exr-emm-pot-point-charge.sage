# Exercise exr-emm-pot-point-charge (Pierrus, Question 8.3): a static charge described by a growing vector potential.
t, x, y, z = var("t x y z")
X = [x, y, z]
def div(F): return sum(diff(F[i], X[i]) for i in range(3)).simplify_full()
def curl(F): return vector([diff(F[2], y) - diff(F[1], z), diff(F[0], z) - diff(F[2], x), diff(F[1], x) - diff(F[0], y)]).simplify_full()
def grad(f): return vector([diff(f, v) for v in X])
q = var('q'); r = sqrt(x^2 + y^2 + z^2)
A = -q * t / (4 * pi) * vector([x, y, z]) / r^3
print('E = -dA/dt =', (-diff(A, t)).simplify_full(), '; B = curl A =', curl(A))
lam = q * t / (4 * pi * r)
print('A - grad(lambda) with lambda = q t/(4 pi r):', (A - grad(lam)).simplify_full(), '; q/(4 pi r) - d lambda/dt =', (q / (4 * pi * r) - diff(lam, t)).simplify_full())

# Exercise exr-emw-pw-packet: a Gaussian wave packet in vacuum.
t, x, y, z = var("t x y z")
X = [x, y, z]
def div(F): return sum(diff(F[i], X[i]) for i in range(3)).simplify_full()
def curl(F): return vector([diff(F[2], y) - diff(F[1], z), diff(F[0], z) - diff(F[2], x), diff(F[1], x) - diff(F[0], y)]).simplify_full()
def grad(f): return vector([diff(f, v) for v in X])
def maxwell(E, B): return [div(E), div(B), (curl(E) + diff(B, t)).simplify_full(), (curl(B) - diff(E, t)).simplify_full()]
E0, sig, k0 = var('E0 sigma k0', domain='positive')
g = function('g')
E = vector([g(z - t), 0, 0]); B = vector([0, g(z - t), 0])
print('vacuum equations for any profile g(z - t):', maxwell(E, B))
f = E0 * exp(-(z - t)^2 / (2 * sig^2)) * cos(k0 * (z - t))
print('the packet is of this form: f depends on z - t only:', bool((diff(f, t) + diff(f, z)).simplify_full() == 0))
u = var('u')
energy = integrate((E0 * exp(-u^2 / (2 * sig^2)) * cos(k0 * u))^2, u, -oo, oo).simplify_full()
print('integral of E_x^2 dz =', energy, '; for k0 sigma >> 1 it tends to E0^2 sigma sqrt(pi)/2 =', E0^2 * sig * sqrt(pi) / 2)

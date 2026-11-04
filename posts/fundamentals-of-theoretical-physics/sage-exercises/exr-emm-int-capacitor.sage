# Exercise exr-emm-int-capacitor: the charging capacitor, fields, potentials and energy balance.
t, x, y, z = var('t x y z')
s, r, I, a, d = var('s r I a d', domain='positive')
X = [x, y, z]
def curl(F): return vector([diff(F[2], y) - diff(F[1], z), diff(F[0], z) - diff(F[2], x), diff(F[1], x) - diff(F[0], y)]).simplify_full()
def grad(f): return vector([diff(f, v) for v in X])
E = I * t / (pi * a^2)
phi = -E * z; A = vector([0, 0, -I * (x^2 + y^2) / (4 * pi * a^2)])
Ef = (-grad(phi) - diff(A, t)).simplify_full(); Bf = curl(A)
print('E =', Ef, '; B =', Bf, '; |B| at distance s: I s/(2 pi a^2)')
print('Ampere-Maxwell inside (no current): curl B - dE/dt =', (curl(Bf) - diff(Ef, t)).simplify_full())
flux = 2 * pi * a * d * E * I * a / (2 * pi * a^2)
energy = integrate((E^2 + (I * r / (2 * pi * a^2))^2) / 2 * 2 * pi * r * d, r, 0, a)
print('inward Poynting flux through the rim:', flux.simplify_full(), '; rate of change of the energy (electric part dominant):', diff(energy, t).simplify_full())

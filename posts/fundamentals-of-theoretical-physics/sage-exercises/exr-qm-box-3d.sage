# Exercise exr-qm-box-3d: energy levels n1^2 + n2^2 + n3^2 (units E_1) of the cubic box and their degeneracies.
from collections import Counter
c = Counter(a * a + b * b + d * d for a in range(1, 8) for b in range(1, 8) for d in range(1, 8))
levels = sorted(c)[:8]
print("lowest eight levels / E_1 and degeneracies:", [(e, c[e]) for e in levels])
print("states of level 14:", [(a, b, d) for a in range(1, 4) for b in range(1, 4) for d in range(1, 4) if a * a + b * b + d * d == 14])
x, y, z = var("x y z"); hbar, m, L = var("hbar m L", domain="positive")
ph = lambda n, u: sqrt(2 / L) * sin(n * pi * u / L)
f = ph(1, x) * ph(2, y) * ph(3, z)
print("-hbar^2/2m Laplacian of phi_1 phi_2 phi_3 = 14 E_1 phi:", bool((-hbar ** 2 / (2 * m) * (diff(f, x, 2) + diff(f, y, 2) + diff(f, z, 2)) - 14 * pi ** 2 * hbar ** 2 / (2 * m * L ** 2) * f).simplify_full() == 0))

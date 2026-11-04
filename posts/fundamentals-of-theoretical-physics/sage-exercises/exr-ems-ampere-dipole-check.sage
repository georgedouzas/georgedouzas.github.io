# Exercise exr-ems-ampere-dipole-check: the dipole field has zero divergence and zero curl for r > 0.
x, y, z, m1, m2, m3 = var('x y z m1 m2 m3')
X = vector([x, y, z]); m = vector([m1, m2, m3]); r = sqrt(x^2 + y^2 + z^2)
B = (3 * m.dot_product(X) * X / r^2 - m) / (4 * pi * r^3)
div = sum(diff(B[i], v) for i, v in enumerate([x, y, z])).simplify_full()
curl = vector([diff(B[2], y) - diff(B[1], z), diff(B[0], z) - diff(B[2], x), diff(B[1], x) - diff(B[0], y)]).simplify_full()
print("div B =", div, "; curl B =", curl)
psi = -m.dot_product(X) / (4 * pi * r^3)
print("B = grad(psi) with psi = -m.x/(4 pi r^3):", bool((B - vector([diff(psi, v) for v in [x, y, z]])).simplify_full() == 0))

# Exercise exr-ems-mech-charge-dipole: force grad(p.E) on a dipole p at position r in the field of a charge q at the origin.
x, y, z, q, p1, p2, p3 = var('x y z q p1 p2 p3')
X = vector([x, y, z]); p = vector([p1, p2, p3]); r = sqrt(x^2 + y^2 + z^2)
E = q * X / (4 * pi * r^3)
F = vector([diff(p.dot_product(E), v) for v in [x, y, z]])
F_expected = q / (4 * pi * r^3) * (p - 3 * p.dot_product(X) * X / r^2)
print("grad(p.E) - q/(4 pi r^3)(p - 3 (p.n) n) =", (F - F_expected).simplify_full())
print("(p.grad)E equals grad(p.E):", bool((F - vector([sum(p[j] * diff(E[i], v) for j, v in enumerate([x, y, z])) for i in range(3)])).simplify_full() == 0))

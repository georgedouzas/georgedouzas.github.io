# Exercise exr-ems-square-axis: four charges q at (+-a/2, +-a/2, 0); the Coulomb field on the axis (Heaviside-Lorentz units).
q, a, z = var('q a z', domain='positive')
X = vector([0, 0, z])
E = sum(q * (X - vector([sx * a / 2, sy * a / 2, 0])) / (X - vector([sx * a / 2, sy * a / 2, 0])).norm()^3 / (4 * pi) for sx in (1, -1) for sy in (1, -1))
E = E.simplify_full()
print("E =", E)
Ez = E[2]
print("E_z equals q z/(pi (z^2 + a^2/2)^(3/2)):", bool((Ez - q * z / (pi * (z^2 + a^2 / 2)^(3/2))).simplify_full() == 0))
crit = solve(diff(Ez, z) == 0, z)
print("critical heights:", crit)

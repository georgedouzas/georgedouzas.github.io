# Exercise exr-ems-ampere-solenoid-integral: the axial field of a finite solenoid integrated over the whole axis.
n, I, a, L, z = var('n I a L z', domain='positive')
Bz = n * I / 2 * ((L / 2 - z) / sqrt(a^2 + (L / 2 - z)^2) + (L / 2 + z) / sqrt(a^2 + (L / 2 + z)^2))
F = integrate(Bz, z).simplify_full()
print("antiderivative:", F)
print("integral over the axis:", (limit(F, z=oo) - limit(F, z=-oo)).simplify_full())

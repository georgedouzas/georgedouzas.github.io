# Exercise exr-ems-mech-stress (Maggiore, Sec. 4.1.7): force on a charge from the Maxwell stress over the midplane.
q, d, s = var('q d s', domain='positive')
E = 2 * q * s / (4 * pi * (s^2 + d^2)^(3/2))       # radial in the plane, no z component
Fz = integrate(E^2 / 2 * 2 * pi * s, s, 0, oo)
print("F_z =", Fz.simplify_full(), "; Coulomb q^2/(4 pi (2d)^2) =", q^2 / (4 * pi * (2 * d)^2), "; equal:", bool((Fz - q^2 / (16 * pi * d^2)).simplify_full() == 0))

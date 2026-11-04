# Exercise exr-fld-dispersion-nonrel: the non-relativistic limit of the Klein-Gordon dispersion relation.
# omega = sqrt(k^2 + m^2) = m + k^2/(2m) - k^4/(8 m^3) + ...: rest energy, kinetic energy, relativistic correction.
# Writing phi = e^{-i m t} psi and dropping d_t^2 psi turns (Box + m^2) phi = 0 into the free Schrodinger equation.
k, m, t, x = var("k m t x", domain="real"); assume(m > 0)
print("omega(k) for k << m:", taylor(sqrt(k**2 + m**2), k, 0, 6))
psi = function("psi")(t, x)
phi = exp(-I * m * t) * psi
kg = (diff(phi, t, 2) - diff(phi, x, 2) + m**2 * phi) * exp(I * m * t)
print("e^{imt} (Box + m^2)(e^{-imt} psi) =", kg.simplify_full())
schr = -2 * m * (I * diff(psi, t) + diff(psi, x, 2) / (2 * m))
print("equals d_t^2 psi - 2m [i d_t psi + d_x^2 psi/(2m)]:", bool((kg - diff(psi, t, 2) - schr).simplify_full() == 0))
print("Schrodinger plane wave e^{i(kx - Et)} needs E =", solve(I * (-I * var("E")) - k**2 / (2 * m) == 0, var("E")))

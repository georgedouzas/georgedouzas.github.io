# Exercise exr-qm-momentum-wavefunction: the momentum wave function of the Gaussian with mean momentum p0, hbar symbolic.
x, p, p0 = var("x p p0", domain="real"); hbar, s = var("hbar s", domain="positive")
psi = (2 * pi * s ** 2) ** (-1 / 4) * exp(-x ** 2 / (4 * s ** 2) + I * p0 * x / hbar)
pt = (integrate(exp(-I * p * x / hbar) * psi, x, -oo, oo) / sqrt(2 * pi * hbar))
expected = (2 * s ** 2 / (pi * hbar ** 2)) ** (1 / 4) * exp(-s ** 2 * (p - p0) ** 2 / hbar ** 2)
print("psi~(p) / expected =", (pt / expected).canonicalize_radical())
rho = expected ** 2
print("norm:", integrate(rho, p, -oo, oo).simplify_full())
print("mean:", integrate(p * rho, p, -oo, oo).simplify_full())
print("variance:", integrate((p - p0) ** 2 * rho, p, -oo, oo).simplify_full())

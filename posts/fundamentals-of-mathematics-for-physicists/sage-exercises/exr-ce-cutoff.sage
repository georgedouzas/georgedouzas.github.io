# Exercise exr-ce-cutoff: the cutoff expansion of the Dirichlet Casimir energy E_eps = (pi/2L) sum n e^{-eps n pi/L}, and
# the Gaussian cutoff, whose finite part is again -pi/24 at L = 1.
e, L = var("epsilon L", domain="positive"); n = var("n", domain="integer")
E = (pi / (2 * L)) * sum(n * exp(-e * n * pi / L), n, 1, oo)
print("E_eps =", taylor(E, e, 0, 2).simplify_full().expand())
RF = RealField(100)
for eps in [RF(1) / 10, RF(1) / 20]:
    Eg = RF(pi) / 2 * sum(k * exp(-eps ** 2 * k ** 2) for k in range(1, 2000))
    print("Gaussian cutoff eps = %s: E - pi/(4 eps^2) = %.12f ; -pi/24 = %.12f" % (eps.n(digits=3), Eg - RF(pi) / (4 * eps ** 2), -RF(pi) / 24))

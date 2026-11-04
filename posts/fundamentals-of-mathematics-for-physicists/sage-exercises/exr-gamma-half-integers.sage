# Exercise exr-gamma-half-integers: Gamma(n + 1/2) = (2n)! sqrt(pi)/(4^n n!) for n = 0..6, exactly.
for n in range(7):
    formula = factorial(2*n)*sqrt(pi)/(4^n*factorial(n))
    print("n = %d: Gamma(n + 1/2) = %s, formula = %s, equal: %s" % (n, gamma(n + 1/2), formula, bool(gamma(n + 1/2) == formula)))

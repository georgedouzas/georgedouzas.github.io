# Exercise exr-st-unbiased-sage: symbolic expectation of polynomials in independent X_1..X_n with E X = mu,
# E X^2 = sigma^2 + mu^2: E xbar = mu, E xbar^2 = mu^2 + sigma^2/n and E s^2 = sigma^2 for n = 2, 3, 6.
mu, sigma = var("mu sigma")
moments = {0: 1, 1: mu, 2: sigma ** 2 + mu ** 2}
def expectation(poly):
    total = 0
    for mono, coeff in poly.dict().items():
        term = coeff
        for e in mono:
            term *= moments[e]
        total += term
    return SR(total).expand()
for n in [2, 3, 6]:
    R = PolynomialRing(SR, ["x%d" % i for i in range(n)])
    xs = R.gens()
    xbar = sum(xs) / n
    s2 = sum((xi - xbar) ** 2 for xi in xs) / (n - 1)
    print("n = %d: E xbar = %s, E xbar^2 = %s, E s^2 = %s" % (n, expectation(xbar), expectation(xbar ** 2).simplify_full(), expectation(s2).simplify_full()))
    print("        E xbar^2 - mu^2 = sigma^2/n:", bool((expectation(xbar ** 2) - mu ** 2 - sigma ** 2 / n).simplify_full() == 0), "; E s^2 = sigma^2:", bool((expectation(s2) - sigma ** 2).simplify_full() == 0))

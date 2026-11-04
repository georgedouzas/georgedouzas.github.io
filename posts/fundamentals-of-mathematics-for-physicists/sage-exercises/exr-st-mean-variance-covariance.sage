# Exercise exr-st-mean-variance-covariance: Cov(xbar, s^2) by symbolic expectation with third moments, for n = 3, 4;
# equals kappa_3/n with kappa_3 = E(X - mu)^3 = mu3 - 3 mu sigma^2 - mu^3.
mu, sigma, mu3 = var("mu sigma mu3")
moments = {0: 1, 1: mu, 2: sigma ** 2 + mu ** 2, 3: mu3}
def expectation(poly):
    total = 0
    for mono, coeff in poly.dict().items():
        term = coeff
        for e in mono:
            term *= moments[e]
        total += term
    return SR(total).expand()
kappa3 = mu3 - 3 * mu * (sigma ** 2 + mu ** 2) + 2 * mu ** 3
for n in [3, 4]:
    R = PolynomialRing(SR, ["x%d" % i for i in range(n)])
    xs = R.gens()
    xbar = sum(xs) / n
    s2 = sum((xi - xbar) ** 2 for xi in xs) / (n - 1)
    cov = (expectation(xbar * s2) - expectation(xbar) * expectation(s2)).simplify_full()
    print("n = %d: Cov(xbar, s^2) =" % n, cov.factor(), "; equals kappa_3/n:", bool((cov - kappa3 / n).simplify_full() == 0))
print("kappa_3 = E(X - mu)^3 =", kappa3.expand())

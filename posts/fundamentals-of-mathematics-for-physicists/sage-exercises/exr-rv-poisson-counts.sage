# Exercise exr-rv-poisson-counts: Poisson probabilities within one standard deviation sqrt(lambda) of the mean for
# lambda = 2, 10, 100, against the Gaussian 0.6827, and the exact P(N = 0), P(N >= 5) for lambda = 2.
pois = lambda lam, k: exp(-lam) * lam ** k / factorial(k)
for lam in [2, 10, 100]:
    s = sqrt(lam)
    pr = sum(pois(lam, k) for k in range(0, 3 * lam + 50) if abs(k - lam) <= s)
    print("lambda = %d: P(|N - lambda| <= sqrt(lambda)) = %.4f (Gaussian: %.4f)" % (lam, float(pr), float(erf(1 / sqrt(2)))))
p0 = pois(2, 0)
p5 = 1 - sum(pois(2, k) for k in range(5))
print("lambda = 2: P(N = 0) =", p0, "=", numerical_approx(p0, digits=4), "; P(N >= 5) =", p5.simplify_full(), "=", numerical_approx(p5, digits=4))

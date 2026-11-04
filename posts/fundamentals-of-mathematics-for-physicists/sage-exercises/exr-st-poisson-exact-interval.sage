# Exercise exr-st-poisson-exact-interval: the exact 95 percent interval for a Poisson mean given an observed count 3,
# from P(N >= 3 | lo) = 0.025 and P(N <= 3 | hi) = 0.025, against the Gaussian 3 +- 1.96 sqrt(3).
lam = var("lam")
pois_cdf = lambda k, l: sum(exp(-l) * l ** j / factorial(j) for j in range(k + 1))
lo = find_root(1 - pois_cdf(2, lam) - 0.025, 0.01, 3)
hi = find_root(pois_cdf(3, lam) - 0.025, 3, 20)
print("exact interval: [%.3f, %.3f]" % (lo, hi))
print("check: P(N >= 3 | lo) = %.4f, P(N <= 3 | hi) = %.4f" % (1 - float(pois_cdf(2, lo)), float(pois_cdf(3, hi))))
g = 1.96 * sqrt(3.0)
print("Gaussian interval: [%.3f, %.3f]" % (3 - g, 3 + g))

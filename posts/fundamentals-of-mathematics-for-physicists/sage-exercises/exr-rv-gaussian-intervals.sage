# Exercise exr-rv-gaussian-intervals: P(|Z| <= k) for the standard Gaussian via erf and via RealDistribution, and the
# two-sided quantiles for 90, 95 and 99 percent.
G = RealDistribution("gaussian", 1)
for k in [1, 2, 3]:
    via_erf = float(erf(k / sqrt(2)))
    via_cdf = G.cum_distribution_function(k) - G.cum_distribution_function(-k)
    print("P(|Z| <= %d) = %.6f (erf), %.6f (cdf)" % (k, via_erf, via_cdf))
for level in [0.90, 0.95, 0.99]:
    z = G.cum_distribution_function_inv((1 + level) / 2)
    print("P(|Z| <= z) = %.2f at z = %.4f" % (level, z))

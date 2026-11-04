# Exercise exr-pr-simulation-study: 5000 seeded samples from the mixture (1/2) N(-1, 1) + (1/2) N(1, 1): sample mean and
# variance with 95 percent intervals (exact values 0 and 2), a single-Gaussian fit, and the chi-squared test over ten
# equiprobable bins of the fitted Gaussian with 10 - 1 - 2 = 7 degrees of freedom.
import random
from math import sqrt, erf
random.seed(int(99))
N = 5000
xs = [random.gauss(-1 if random.random() < 0.5 else 1, 1) for _ in range(N)]
xbar = sum(xs) / N
s2 = sum((x - xbar) ** 2 for x in xs) / (N - 1)
m4 = sum((x - xbar) ** 4 for x in xs) / N
print("sample mean %.4f +- %.4f (exact 0)" % (xbar, 1.96 * sqrt(s2 / N)))
print("sample variance %.4f +- %.4f (exact 2)" % (s2, 1.96 * sqrt((m4 - s2 ** 2) / N)))
print("sample kurtosis m4/s^4 = %.3f (exact 2.5 for the mixture, 3 for a Gaussian)" % (m4 / s2 ** 2))
G = RealDistribution("gaussian", 1)
edges = [xbar + sqrt(s2) * G.cum_distribution_function_inv(j / 10.0) for j in range(1, 10)]
counts = [0] * 10
for x in xs:
    b = sum(1 for e in edges if x > e)
    counts[b] += 1
expected = N / 10.0
chi2 = sum((c - expected) ** 2 / expected for c in counts)
C7 = RealDistribution("chisquared", 7)
print("bin counts under the fitted Gaussian (expected 500 each):", counts)
print("chi^2 = %.2f with 7 degrees of freedom; 95%% quantile %.2f; p-value %.2e" % (chi2, C7.cum_distribution_function_inv(0.95), 1 - C7.cum_distribution_function(chi2)))
print("single Gaussian rejected at the 5% level:", chi2 > C7.cum_distribution_function_inv(0.95))

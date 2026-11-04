# Exercise exr-lt-monte-carlo-sample-size: Chebyshev's sample size for estimating pi to within 0.01 with confidence
# 0.95 from the fraction of uniform points in the quarter disc, and the actual error of a seeded run of that size.
import random
p = pi / 4
var_est = 16 * p * (1 - p)           # variance of 4 * indicator
N_cheb = ceil(var_est / (QQ(1) / 10000 * QQ(1) / 20))
print("Var(4 I) = 16 p (1 - p) =", numerical_approx(var_est, digits=6))
print("Chebyshev sample size for |error| < 0.01 with probability >= 0.95:", N_cheb)
random.seed(int(2024))
count = 0
for _ in range(N_cheb):
    u, v = random.random(), random.random()
    count += (u * u + v * v <= 1)
est = 4.0 * count / N_cheb
print("estimate %.5f, error %.5f, predicted standard deviation sqrt(Var/N) = %.5f" % (est, abs(est - float(pi)), float(sqrt(var_est / N_cheb))))

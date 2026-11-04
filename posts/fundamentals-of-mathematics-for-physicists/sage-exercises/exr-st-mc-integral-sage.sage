# Exercise exr-st-mc-integral-sage: Monte Carlo estimate of int_0^1 sqrt(1 - x^2) dx = pi/4 from 10^4 seeded uniform
# points with the estimated standard error s/sqrt(N), against the exact tau^2 = 2/3 - pi^2/16.
import random
from math import sqrt as msqrt
random.seed(int(42))
N = 10 ** 4
g = [msqrt(1 - x * x) for x in (random.random() for _ in range(N))]
est = sum(g) / N
s2 = sum((v - est) ** 2 for v in g) / (N - 1)
tau2 = QQ(2) / 3 - pi ** 2 / 16
print("estimate of pi/4: %.5f +- %.5f (exact pi/4 = %.5f, actual error %.5f)" % (est, msqrt(s2 / N), float(pi / 4), abs(est - float(pi / 4))))
print("sample variance of g: %.5f ; exact tau^2 = 2/3 - pi^2/16 = %.5f" % (s2, float(tau2)))
print("variance of the indicator estimate of pi/4: (pi/4)(1 - pi/4) = %.5f, ratio %.2f" % (float(pi / 4 * (1 - pi / 4)), float(pi / 4 * (1 - pi / 4) / tau2)))

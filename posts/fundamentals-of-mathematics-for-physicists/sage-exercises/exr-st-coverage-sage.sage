# Exercise exr-st-coverage-sage: coverage of the interval xbar +- 1.96 s/sqrt(n) for the mean of an exponential
# distribution (mean 1) over 2000 seeded samples of size n = 30 and n = 5.
import random
from math import log, sqrt
random.seed(int(17))
for n in [30, 5]:
    covered = 0
    for _ in range(2000):
        xs = [-log(1 - random.random()) for _ in range(n)]
        xbar = sum(xs) / n
        s = sqrt(sum((x - xbar) ** 2 for x in xs) / (n - 1))
        half = 1.96 * s / sqrt(n)
        covered += (xbar - half <= 1 <= xbar + half)
    print("n = %2d: fraction of intervals containing the true mean 1: %.3f (nominal 0.95)" % (n, covered / 2000))

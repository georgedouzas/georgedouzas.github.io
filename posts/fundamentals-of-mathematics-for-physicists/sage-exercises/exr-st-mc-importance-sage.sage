# Exercise exr-st-mc-importance-sage: int_0^1 e^-x dx = 1 - 1/e estimated uniformly and by importance sampling from
# q(x) = (2/3)(2 - x), sampled by inverting Q(x) = (2/3)(2x - x^2/2); sample variances and standard errors compared.
import random
from math import exp, sqrt
random.seed(int(8))
N = 10 ** 4
exact = 1 - exp(-1)
us = [random.random() for _ in range(N)]
g_uniform = [exp(-u) for u in us]
# inverse of Q(x) = (2/3)(2x - x^2/2): x^2 - 4x + 3u = 0, root in [0, 1]: x = 2 - sqrt(4 - 3u)
q = lambda x: 2.0 / 3 * (2 - x)
g_weighted = [exp(-x) / q(x) for x in (2 - sqrt(4 - 3 * u) for u in us)]
for name, g in [("uniform", g_uniform), ("importance", g_weighted)]:
    est = sum(g) / N
    s2 = sum((v - est) ** 2 for v in g) / (N - 1)
    print("%-10s: estimate %.5f (exact %.5f, error %.5f), sample variance %.5f, standard error %.5f" % (name, est, exact, abs(est - exact), s2, sqrt(s2 / N)))
s2u = sum((v - sum(g_uniform) / N) ** 2 for v in g_uniform) / (N - 1)
s2w = sum((v - sum(g_weighted) / N) ** 2 for v in g_weighted) / (N - 1)
print("variance reduction factor: %.1f" % (s2u / s2w))

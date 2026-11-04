# Exercise exr-lt-clt-simulation: 10^4 seeded samples of S_n = U_1 + ... + U_n for n = 1, 2, 12, standardised with
# mean n/2 and variance n/12; fractions within one and two standard deviations against 0.6827 and 0.9545.
import random
random.seed(int(31))
M = 10 ** 4
for n in [1, 2, 12]:
    mu, sd = n / 2.0, sqrt(n / 12.0)
    zs = [(sum(random.random() for _ in range(n)) - mu) / sd for _ in range(M)]
    f1 = sum(1 for z in zs if abs(z) <= 1) / float(M)
    f2 = sum(1 for z in zs if abs(z) <= 2) / float(M)
    print("n = %2d: within 1 sd %.4f (Gaussian 0.6827), within 2 sd %.4f (Gaussian 0.9545)" % (n, f1, f2))
print("exact for n = 1: P(|U - 1/2| <= 1/sqrt 12) = 2/sqrt(12) = %.4f, within 2 sd: 1" % float(2 / sqrt(12)))

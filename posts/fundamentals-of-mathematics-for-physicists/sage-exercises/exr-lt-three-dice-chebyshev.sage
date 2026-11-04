# Exercise exr-lt-three-dice-chebyshev: exact tails P(|S_3 - 21/2| >= k sigma) for the sum of three dice
# (sigma^2 = 35/4) against Chebyshev's 1/k^2 and the Gaussian 1 - erf(k/sqrt 2), for k = 1, 3/2, 2.
import itertools
Omega = list(itertools.product(range(1, 7), repeat=3))
P = lambda E: QQ(len(E)) / len(Omega)
mu, sigma = QQ(21) / 2, sqrt(QQ(35) / 4)
mean = sum(sum(w) for w in Omega) / QQ(len(Omega)); var = sum(sum(w) ** 2 for w in Omega) / QQ(len(Omega)) - mean ** 2
print("mean", mean, ", variance", var, "= 35/4:", var == QQ(35) / 4)
for k in [1, QQ(3) / 2, 2]:
    tail = P([w for w in Omega if abs(sum(w) - mu) >= k * sigma])
    print("k = %.1f: exact tail %.4f (= %s), Chebyshev %.4f, Gaussian %.4f" % (float(k), float(tail), tail, float(1 / k ** 2), float(1 - erf(k / sqrt(2)))))

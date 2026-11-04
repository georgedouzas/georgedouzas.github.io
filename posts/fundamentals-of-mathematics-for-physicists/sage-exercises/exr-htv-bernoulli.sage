# Exercise exr-htv-bernoulli: zeta(-n) = -B_{n+1}/(n + 1), zeta(2n) = (-1)^(n+1) B_{2n} (2 pi)^{2n} / (2 (2n)!), and the
# trivial zeros zeta(-2n) = 0.
print("zeta(-n) = -B_{n+1}/(n+1) for n = 1..9:", all(zeta(-k) == -bernoulli(k + 1) / (k + 1) for k in srange(1, 10)), [zeta(-k) for k in srange(1, 10)])
print("zeta(2n) formula for n = 1..4:", all(bool(zeta(2 * k) == (-1) ** (k + 1) * bernoulli(2 * k) * (2 * pi) ** (2 * k) / (2 * factorial(2 * k))) for k in srange(1, 5)), [zeta(2 * k) for k in srange(1, 5)])
print("zeta(-2n) = 0 for n = 1..4:", all(zeta(-2 * k) == 0 for k in srange(1, 5)))

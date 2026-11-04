# Exercise exr-fan-bernstein-corner: the error B_n f(1/2) - f(1/2) for f(x) = |x - 1/2| equals E|S_n/n - 1/2| for a
# binomial S_n(1/2); exact values and comparison with 1/sqrt(2 pi n).
def err(n):
    return sum(binomial(n, k) * QQ(1) / 2^n * abs(QQ(k) / n - QQ(1) / 2) for k in range(n + 1))
print("n = 2: error", err(2))
for n in (2, 10, 50, 200, 1000):
    e = err(n)
    print("n = %4d: error %.6f, 1/sqrt(2 pi n) = %.6f, ratio %.4f" % (n, e, 1 / sqrt(2 * pi.n() * n), e * sqrt(2 * pi.n() * n)))

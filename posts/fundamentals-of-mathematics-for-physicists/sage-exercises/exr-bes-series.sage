# Exercise exr-bes-series: the truncated Bessel series (K = 6) in Bessel's equation for n = 0, 1, 2; the residual is the
# single monomial (-1)^K / (K! (K + n)!) x^(2K + n + 2) / 2^(2K + n).
x = var("x")
K = 6
for n in range(3):
    J = sum((-1) ** k / (factorial(k) * factorial(k + n)) * (x / 2) ** (2 * k + n) for k in range(K + 1))
    res = (x ** 2 * diff(J, x, 2) + x * diff(J, x) + (x ** 2 - n ** 2) * J).expand()
    predicted = (-1) ** K / (factorial(K) * factorial(K + n)) * x ** (2 * K + n + 2) / 2 ** (2 * K + n)
    print("n =", n, ": residual =", res, "| equals the predicted monomial:", bool((res - predicted).expand() == 0))

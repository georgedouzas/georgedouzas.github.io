# Exercise exr-limit-rules: limits computed with Sage's limit, each compared with the value predicted by the limit
# rules (quotients of polynomials, the squeeze n sin(1/n) -> 1, the growth (n!)^(1/n)/n -> 1/e).
n = var("n")
cases = [((3 * n ** 2 + 1) / (n ** 2 - 5), 3), (n * sin(1 / n), 1), ((n ** 2 + n) ** (1 / 2) - n, QQ(1) / 2), (factorial(n) ** (1 / n) / n, exp(-1))]
for expr, expected in cases:
    L = limit(expr, n=oo)
    print("lim", expr, "=", L, "; expected", expected, ":", bool(L == expected))

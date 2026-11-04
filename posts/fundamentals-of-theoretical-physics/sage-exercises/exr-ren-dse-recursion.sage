# Exercise exr-ren-dse-recursion: the zero-dimensional Dyson-Schwinger recursion m_(k+1) = (2k+1) m_k - (lam/6) m_(k+2),
# m_k = <phi^(2k)>, solved order by order in lam, compared with the direct expansion of the ratio of integrals.
N = 6
R.<lam> = PowerSeriesRing(QQ, default_prec=N + 1)
K = N + 3
coef = [[0] * (N + 1) for _ in range(2 * K + 4)]     # coef[k][j] = [lam^j] m_k
for k in range(2 * K + 4):
    coef[k][0] = prod(range(2 * k - 1, 0, -2))
for j in range(1, N + 1):
    # m_(k+1) = (2k+1) m_k - (1/6) [lam^(j-1)] m_(k+2); m_0 = 1 exactly
    for k in range(0, 2 * K + 1 - j):
        coef[k + 1][j] = (2 * k + 1) * coef[k][j] - QQ(1/6) * coef[k + 2][j - 1]
print("<phi^2> from the recursion:", [coef[1][j] for j in range(N + 1)])
var('l')
mom = lambda k: 0 if k % 2 else prod(range(k - 1, 0, -2))
num = sum((-l / 24)^j / factorial(j) * mom(2 + 4 * j) for j in range(N + 1)); den = sum((-l / 24)^j / factorial(j) * mom(4 * j) for j in range(N + 1))
print("<phi^2> directly:           ", [taylor(num / den, l, 0, N).coefficient(l, j) for j in range(N + 1)])

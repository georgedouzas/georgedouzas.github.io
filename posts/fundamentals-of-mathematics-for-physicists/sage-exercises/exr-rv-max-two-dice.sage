# Exercise exr-rv-max-two-dice: the probability mass function and distribution function of the maximum of two dice,
# by enumeration, against the formulas F(m) = m^2/36 and p_m = (2m - 1)/36; comparison with P(S = 7).
Omega = [(i, j) for i in range(1, 7) for j in range(1, 7)]
P = lambda E: QQ(len(E)) / len(Omega)
pmf = {m: P([w for w in Omega if max(w) == m]) for m in range(1, 7)}
cdf = {m: P([w for w in Omega if max(w) <= m]) for m in range(1, 7)}
print("pmf of the maximum:", pmf)
print("matches (2m - 1)/36:", all(pmf[m] == QQ(2 * m - 1) / 36 for m in range(1, 7)))
print("distribution function:", cdf)
print("matches m^2/36:", all(cdf[m] == QQ(m * m) / 36 for m in range(1, 7)))
print("P(M = 6) =", pmf[6], "; P(S = 7) =", P([w for w in Omega if sum(w) == 7]))
print("E[M] =", sum(m * p for m, p in pmf.items()))

# Exercise exr-rv-binomial-variance: variances from exact tables: the sum of two dice has variance 35/6 and the
# number of heads in 10 flips has variance 10/4 = 5/2, against the formulas 2 * 35/12 and np(1 - p).
Omega = [(i, j) for i in range(1, 7) for j in range(1, 7)]
P = lambda E: QQ(len(E)) / len(Omega)
pmf = {s: P([w for w in Omega if sum(w) == s]) for s in range(2, 13)}
mean = sum(s * p for s, p in pmf.items()); vr = sum(s ** 2 * p for s, p in pmf.items()) - mean ** 2
print("two dice: mean", mean, ", variance", vr, "= 2 * 35/12:", vr == 2 * QQ(35) / 12)
n, p = 10, QQ(1) / 2
pm = {j: binomial(n, j) * p ** j * (1 - p) ** (n - j) for j in range(n + 1)}
m = sum(j * q for j, q in pm.items()); v = sum(j ** 2 * q for j, q in pm.items()) - m ** 2
print("10 flips: mean", m, ", variance", v, "= np(1 - p):", v == n * p * (1 - p))
pa = var("pa")
print("one indicator with P(A) = pa: variance E[1_A^2] - P(A)^2 =", (pa - pa ** 2).factor())

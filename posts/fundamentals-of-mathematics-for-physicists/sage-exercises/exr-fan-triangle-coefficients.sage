# Exercise exr-fan-triangle-coefficients: Fourier coefficients of |x| on [-pi, pi] and its H^1 norm from Parseval.
x = var('x'); k = var('k', domain='integer'); assume(k > 0)
ck = (integrate(x * cos(k * x), x, 0, pi) / pi).simplify_full()
print("c_k (k != 0) =", ck, "; c_0 =", integrate(x, x, 0, pi) / pi)
L2 = 2 * pi * ((pi / 2)^2 + 2 * sum((2 / (pi * m^2))^2 for m in range(1, 4001, 2)))
D2 = 2 * pi * 2 * sum(m^2 * (2 / (pi * m^2))^2 for m in range(1, 4001, 2))
print("2 pi sum |c_k|^2 = %.6f vs 2 pi^3/3 = %.6f" % (L2.n(), (2 * pi^3 / 3).n()))
print("2 pi sum k^2 |c_k|^2 = %.6f vs 2 pi = %.6f (partial sum up to |k| = 4000)" % (D2.n(), (2 * pi).n()))
print("exact sums over odd k > 0: sum 1/k^2 =", sum(1 / (2 * var('j') + 1)^2, var('j'), 0, oo), ", sum 1/k^4 =", sum(1 / (2 * var('j') + 1)^4, var('j'), 0, oo))

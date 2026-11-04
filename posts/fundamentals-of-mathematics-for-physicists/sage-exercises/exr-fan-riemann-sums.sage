# Exercise exr-fan-riemann-sums: mu_n(f) = (1/n) sum_{k=1}^n f(k/n) on polynomials; the exact error for x^2.
n, k = var('n k'); assume(n > 0)
e2 = (sum(k^2, k, 1, n) / n^3 - 1 / 3).simplify_full()
print("mu_n(x^2) - 1/3 =", e2.expand())
for m in (1, 2, 3, 5):
    print("x^%d: mu_n(x^%d) - 1/(%d+1) =" % (m, m, m), (sum(k^m, k, 1, n) / n^(m + 1) - 1 / (m + 1)).simplify_full().expand())
print("non-polynomial test f = exp(x), n = 10, 100, 1000:", [(sum(exp(QQ(j) / nv) for j in range(1, nv + 1)) / nv - (e - 1)).n(digits=5) for nv in (10, 100, 1000)])

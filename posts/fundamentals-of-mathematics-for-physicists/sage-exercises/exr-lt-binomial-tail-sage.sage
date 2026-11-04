# Exercise exr-lt-binomial-tail-sage: exact P(|mean of n flips - 1/2| >= 1/10) against Chebyshev's 25/n and the
# two-sided Chernoff-Hoeffding bound 2 exp(-2 n (1/10)^2), for n = 25, 100, 400.
for n in [25, 100, 400]:
    pm = {j: binomial(n, j) / QQ(2) ** n for j in range(n + 1)}
    tail = sum(q for j, q in pm.items() if abs(QQ(j) / n - QQ(1) / 2) >= QQ(1) / 10)
    cheb = QQ(25) / n
    chern = 2 * exp(-2 * n * QQ(1) / 100)
    print("n = %3d: exact %.6f, Chebyshev %.4f, Chernoff %.5f" % (n, float(tail), float(cheb), float(chern)))

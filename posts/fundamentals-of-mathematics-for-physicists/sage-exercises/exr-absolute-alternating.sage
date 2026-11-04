# Exercise exr-absolute-alternating: the alternating harmonic series converges to log 2 but not absolutely (the
# harmonic series diverges); the partial sums alternate around the limit with error below the first omitted term.
n = var("n")
print("sum (-1)^(n+1)/n =", sum((-1) ** (n + 1) / n, n, 1, oo))
s = lambda m: sum((-1) ** (k + 1) / QQ(k) for k in range(1, m + 1))
for m in [10, 11, 100, 101]:
    err = s(m) - log(2)
    print("m =", m, ": s_m - log 2 =", numerical_approx(err, digits=6), "; |error| < 1/(m+1):", bool(abs(err) < QQ(1) / (m + 1)), "; sign", "+" if err > 0 else "-")
print("the series of absolute values is the harmonic series, whose partial sums H_100 =", numerical_approx(sum(1 / QQ(k) for k in range(1, 101)), digits=6), "exceed any bound as m grows")

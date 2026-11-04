# Exercise exr-basel-numeric: the partial sums of sum 1/n^2 approach pi^2/6 with an error close to 1/N (the tail is
# between 1/(N+1) and 1/N), so a million terms give only six correct digits.
S = lambda N: sum(1 / QQ(k) ** 2 for k in range(1, N + 1))
for N in [10, 100, 1000]:
    err = pi ** 2 / 6 - S(N)
    print("N =", N, ": s_N =", numerical_approx(S(N), digits=10), "; tail pi^2/6 - s_N =", numerical_approx(err, digits=6), "; N * tail =", numerical_approx(N * err, digits=5))
print("pi^2/6 =", numerical_approx(pi ** 2 / 6, digits=12))

# Exercise exr-euler-constant: the sequence H_n - log n, with H_n the n-th harmonic number, decreases to Euler's
# constant gamma = 0.5772...; the differences to gamma behave like 1/(2n).
H = lambda m: sum(1 / QQ(k) for k in range(1, m + 1))
for m in [10, 100, 1000, 10000]:
    d = H(m) - log(m) - euler_gamma
    print("n =", m, ": H_n - log n =", numerical_approx(H(m) - log(m), digits=10), "; difference to gamma =", numerical_approx(d, digits=6), "; times 2n:", numerical_approx(2 * m * d, digits=6))
print("gamma =", numerical_approx(euler_gamma, digits=12))

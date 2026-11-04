# Exercise exr-fs-leibniz: Leibniz's series pi/4 = 1 - 1/3 + 1/5 - ... from the square wave at x = pi/2, with the slow
# convergence of its partial sums.
n = var("n")
print("sum (-1)^n/(2n+1) =", sum((-1) ** n / (2 * n + 1), n, 0, oo))
for N in [10, 100, 1000]:
    s = sum((-1) ** k / QQ(2 * k + 1) for k in range(N))
    print("N =", N, ": 4 s_N =", (4 * s).n(digits=8), "; error |4 s_N - pi| =", (abs(4 * s - pi)).n(digits=3), "; about 1/N:", bool(0.5 / N < abs(4 * s - pi) < 2 / N))

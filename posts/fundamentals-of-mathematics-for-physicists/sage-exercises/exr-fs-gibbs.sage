# Exercise exr-fs-gibbs: the square-wave partial sum S_N at its first maximum x = pi/(N + 1) for growing N, compared
# with the Gibbs constant (2/pi) Si(pi) = 1.17897974...
gibbs = (2 / pi * sin_integral(pi)).n(digits=12)
print("Gibbs constant (2/pi) Si(pi) =", gibbs)
prev = None
for N in [9, 49, 199, 999]:
    xN = pi / (N + 1)
    SN = sum(4 / (pi * n) * sin(n * xN) for n in range(1, N + 1, 2)).n(digits=12)
    d = SN - gibbs
    ratio = ("" if prev is None else "; ratio of differences %.2f" % (prev / d))
    print("N =", N, ": S_N(pi/(N+1)) =", SN, "; difference from the limit =", d.n(digits=4), "; above the limit:", bool(d > 0), ratio)
    prev = d
print("N^2 times the difference is roughly constant (Riemann-sum error of order 1/N^2):", bool(abs((999 ** 2 * (sum(4 / (pi * n) * sin(n * pi / 1000) for n in range(1, 1000, 2)).n(digits=12) - gibbs)) / (199 ** 2 * (sum(4 / (pi * n) * sin(n * pi / 200) for n in range(1, 200, 2)).n(digits=12) - gibbs)) - 1) < 0.1))

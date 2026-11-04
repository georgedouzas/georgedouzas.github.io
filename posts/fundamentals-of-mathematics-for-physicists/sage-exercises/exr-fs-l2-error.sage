# Exercise exr-fs-l2-error: L^2 errors int (f - S_N)^2 of the square-wave partial sums, by exact integration and by
# Parseval's tail sum, with the asymptotic 8/(pi N).
x = var("x")
S = lambda N: sum(4 / (pi * n) * sin(n * x) for n in range(1, N + 1, 2))
for N in [1, 3, 9, 29]:
    exact = 2 * integrate((1 - S(N)) ** 2, x, 0, pi)
    tail = 2 * pi - pi * sum((4 / (pi * n)) ** 2 for n in range(1, N + 1, 2))
    print("N =", N, ": exact error =", exact.n(digits=6), "; Parseval tail =", tail.n(digits=6), "; equal:", bool((exact - tail).expand() == 0), "; N * error =", (N * exact).n(digits=4))
print("8/pi =", (8 / pi).n(digits=4))

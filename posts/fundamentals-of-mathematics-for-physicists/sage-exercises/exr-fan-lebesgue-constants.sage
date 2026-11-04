# Exercise exr-fan-lebesgue-constants: L_N = (1/2pi) int |D_N| with D_N(x) = sin((N + 1/2)x)/sin(x/2), compared with
# L_1 = 1/3 + 2 sqrt(3)/pi and the lower bound (4/pi^2) sum_{k <= N} 1/k.
def L(N):
    D = lambda x: 1 + 2 * sum(cos(k * x) for k in range(1, N + 1))
    # integrate |D| over [0, pi] piecewise between the zeros 2 pi j/(2N + 1)
    zs = [0] + [2 * pi.n() * j / (2 * N + 1) for j in range(1, N + 1)] + [pi.n()]
    return 2 * sum(abs(numerical_integral(D, a, b)[0]) for a, b in zip(zs, zs[1:])) / (2 * pi.n())
print("L_0 = %.6f, L_1 = %.6f, 1/3 + 2 sqrt(3)/pi = %.6f" % (L(0), L(1), (1 / 3 + 2 * sqrt(3) / pi).n()))
for N in (2, 4, 8, 16, 32, 64):
    print("N = %2d: L_N = %.4f, lower bound %.4f, (4/pi^2) log N + 1.27 = %.4f" % (N, L(N), 4 / pi.n()^2 * sum(1 / k for k in range(1, N + 1)), 4 / pi.n()^2 * log(N) + 1.27))

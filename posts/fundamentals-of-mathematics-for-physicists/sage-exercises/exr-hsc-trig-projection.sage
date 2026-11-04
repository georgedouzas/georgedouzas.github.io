# Exercise exr-hsc-trig-projection: the projection of sign(x) on (-pi, pi) onto the modes e^{inx}/sqrt(2 pi), |n| <= N,
# for N = 1, 3, 5, the errors ||f - P_N f||^2 by direct integration and by Parseval.
x = var("x", domain="real")
f = sign(x)
def coef(n):      # <e_n, f> with e_n = e^{inx}/sqrt(2 pi), computed on (0, pi) and (-pi, 0) separately
    return (integrate(exp(-I * n * x), x, 0, pi) - integrate(exp(-I * n * x), x, -pi, 0)) / sqrt(2 * pi)
norm2 = 2 * pi
for N in [1, 3, 5]:
    P = sum(coef(n) * exp(I * n * x) / sqrt(2 * pi) for n in range(-N, N + 1)).simplify_full()
    err_direct = (integrate((1 - P) ** 2, x, 0, pi) + integrate((-1 - P) ** 2, x, -pi, 0)).simplify_full()
    err_parseval = (norm2 - sum(abs(coef(n)) ** 2 for n in range(-N, N + 1))).simplify_full()
    print("N =", N, ": P_N f =", P, "; error =", err_direct, "=", err_direct.n(digits=4), "; Parseval:", bool(err_direct == err_parseval))

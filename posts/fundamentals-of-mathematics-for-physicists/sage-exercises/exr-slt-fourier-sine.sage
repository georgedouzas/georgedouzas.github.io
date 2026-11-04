# Exercise exr-slt-fourier-sine: sine coefficients of f = x(pi - x) on [0, pi], the formula b_n = 4(1 - (-1)^n)/(pi n^3),
# and the squared error of the partial sums S_1, S_3, S_5.
x, n = var("x n")
f = x * (pi - x)
b = lambda k: 2 / pi * integrate(f * sin(k * x), x, 0, pi)
print("b_1..b_6:", [b(k).simplify_full() for k in range(1, 7)])
print("formula 4(1 - (-1)^n)/(pi n^3) for n = 1..6:", [(4 * (1 - (-1) ** k) / (pi * k ** 3)) for k in range(1, 7)])
print("agree:", all(bool((b(k) - 4 * (1 - (-1) ** k) / (pi * k ** 3)).simplify_full() == 0) for k in range(1, 7)))
norm2 = integrate(f ** 2, x, 0, pi)
print("||f||^2 =", norm2)
for N in [1, 3, 5]:
    S = sum(b(k) * sin(k * x) for k in range(1, N + 1))
    err = integrate((f - S) ** 2, x, 0, pi).simplify_full()
    print("N =", N, ": ||f - S_N||^2 =", err, "=", numerical_approx(err), "| relative:", numerical_approx(err / norm2))

# Exercise exr-fan-exp-abs-h1: the H^1 norm of f(x) = e^{-|x|} in position space and in Fourier space.
x, xi = var("x xi")
L2 = 2 * integrate(exp(-2 * x), x, 0, oo)
D2 = 2 * integrate(exp(-2 * x), x, 0, oo)
print("||f||^2 =", L2, "  ||f'||^2 =", D2, "  sum =", L2 + D2)
fhat = 2 * integrate(exp(-x) * cos(xi * x), x, 0, oo)
print("Fourier transform:", fhat.simplify_full())
F = integrate((1 + xi ^ 2) * fhat ^ 2, xi, -oo, oo) / (2 * pi)
print("(1/2 pi) int (1 + xi^2) |fhat|^2 =", F.simplify_full())

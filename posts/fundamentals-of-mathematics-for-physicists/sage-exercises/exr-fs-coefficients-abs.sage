# Exercise exr-fs-coefficients-abs: Fourier coefficients of |x| on (-pi, pi): a_0 = pi, a_n = 2(cos(n pi) - 1)/(pi n^2),
# b_n = 0, computed by exact integration over the two pieces.
x = var("x")
a = lambda n: (integrate(-x * cos(n * x), x, -pi, 0) + integrate(x * cos(n * x), x, 0, pi)) / pi
b = lambda n: (integrate(-x * sin(n * x), x, -pi, 0) + integrate(x * sin(n * x), x, 0, pi)) / pi
print("a_0 =", a(0))
for n in range(1, 7):
    print("n =", n, ": a_n =", a(n), " b_n =", b(n), " a_n = -4/(pi n^2) for odd n, 0 for even:", bool(a(n) == (-4 / (pi * n ** 2) if n % 2 else 0)))

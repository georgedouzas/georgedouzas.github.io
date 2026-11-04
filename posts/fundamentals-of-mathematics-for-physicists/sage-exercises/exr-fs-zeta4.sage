# Exercise exr-fs-zeta4: Fourier coefficients of x^2 on (-pi, pi), zeta(2) from the value at x = pi and zeta(4) from
# Parseval's identity.
x, n = var("x n")
assume(n, "integer")
a0 = integrate(x ** 2, x, -pi, pi) / pi
an = (integrate(x ** 2 * cos(n * x), x, -pi, pi) / pi).simplify_full()
bn = integrate(x ** 2 * sin(n * x), x, -pi, pi) / pi
print("a_0 =", a0, "; a_n =", an, "; b_n =", bn)
print("a_n = 4 (-1)^n / n^2 for n = 1..4:", [an.subs(n=k) for k in range(1, 5)])
value_at_pi = a0 / 2 + sum(4 * (-1) ** n / n ** 2 * cos(n * pi), n, 1, oo)
print("series at x = pi:", value_at_pi, "= pi^2:", bool(value_at_pi == pi ** 2), "so zeta(2) = (pi^2 - pi^2/3)/4 =", (pi ** 2 - pi ** 2 / 3) / 4)
lhs = integrate(x ** 4, x, -pi, pi) / pi
rhs = a0 ** 2 / 2 + sum((4 / n ** 2) ** 2, n, 1, oo)
print("Parseval: (1/pi) int x^4 =", lhs, "; a_0^2/2 + sum a_n^2 =", rhs, "; equal:", bool(lhs == rhs))
print("hence zeta(4) = (2 pi^4/5 - 2 pi^4/9)/16 =", ((2 * pi ** 4 / 5 - 2 * pi ** 4 / 9) / 16).simplify_full(), "= pi^4/90:", bool((2 * pi ** 4 / 5 - 2 * pi ** 4 / 9) / 16 == pi ** 4 / 90))

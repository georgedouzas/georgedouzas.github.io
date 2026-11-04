# Exercise exr-ft-sinc-integrals: int sin^2 x/x^2 = pi and int sin^4 x/x^4 = 2 pi/3 by Plancherel for the box and the
# triangle, checked by direct integration.
x, k = var("x k")
i2 = integrate(sin(x) ** 2 / x ** 2, x, -oo, oo); i4 = integrate(sin(x) ** 4 / x ** 4, x, -oo, oo)
print("int sin^2 x / x^2 dx =", i2, "; Plancherel for the box: 2 pi * 2 / 4 =", 2 * pi * 2 / 4)
tri_norm = 2 * integrate((2 - x) ** 2, x, 0, 2)
print("int Lambda_1^2 =", tri_norm, "; int sin^4 x / x^4 dx =", i4, "; Plancherel for the triangle: 2 pi * (16/3) / 16 =", 2 * pi * tri_norm / 16, "; equal:", bool(i4 == 2 * pi * tri_norm / 16))

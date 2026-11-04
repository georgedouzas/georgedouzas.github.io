# Exercise exr-leibniz-rule-sage: differentiation under the integral sign: d/da int_0^oo e^{-a x} dx = -1/a^2 equals
# -int_0^oo x e^{-a x} dx; and the Gaussian moments from d/da of sqrt(pi/a). Also Feynman's trick on
# int_0^1 (x^a - 1)/log x dx = log(1 + a).
x, a = var("x a")
assume(a > 0)
lhs = diff(integrate(exp(-a * x), x, 0, oo), a); rhs = integrate(diff(exp(-a * x), a), x, 0, oo)
print("d/da int e^{-ax} dx =", lhs, "; int d/da e^{-ax} dx =", rhs, "; equal:", bool(lhs == rhs))
G = integrate(exp(-a * x ** 2), x, -oo, oo)
print("Gaussian: G(a) =", G, "; -G'(a) =", (-diff(G, a)).simplify_full(), "= int x^2 e^{-a x^2} dx =", integrate(x ** 2 * exp(-a * x ** 2), x, -oo, oo).simplify_full())
print("Feynman's trick: F(a) = int_0^1 (x^a - 1)/log x dx has F'(a) = int_0^1 x^a dx =", integrate(x ** a, x, 0, 1), "and F(0) = 0, so F(a) = log(1 + a);")
print("numerical check at a = 1:", numerical_integral((x - 1) / log(x), 0, 1)[0], "vs log 2 =", log(2).n(), "; at a = 3:", numerical_integral((x ** 3 - 1) / log(x), 0, 1)[0], "vs log 4 =", log(4).n())

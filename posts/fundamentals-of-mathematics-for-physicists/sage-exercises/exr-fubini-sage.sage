# Exercise exr-fubini-sage: iterated integrals in both orders for x^2 y on [0, 1] x [0, 2] and for e^{x+y} on the
# triangle 0 <= y <= x <= 1 (where the limits of the inner integral depend on the outer variable).
x, y = var("x y")
A1 = integrate(integrate(x ** 2 * y, x, 0, 1), y, 0, 2); A2 = integrate(integrate(x ** 2 * y, y, 0, 2), x, 0, 1)
print("rectangle: dx first then dy:", A1, "; dy first then dx:", A2, "; equal:", bool(A1 == A2))
T1 = integrate(integrate(exp(x + y), y, 0, x), x, 0, 1); T2 = integrate(integrate(exp(x + y), x, y, 1), y, 0, 1)
print("triangle 0 <= y <= x <= 1: inner dy from 0 to x:", T1.simplify_full(), "; inner dx from y to 1:", T2.simplify_full(), "; equal:", bool((T1 - T2).simplify_full() == 0))
print("the triangle's area as a double integral:", integrate(integrate(1, y, 0, x), x, 0, 1))

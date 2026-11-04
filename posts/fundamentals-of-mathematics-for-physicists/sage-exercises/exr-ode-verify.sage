# Exercise exr-ode-verify: verify candidate solutions by substitution. C1 cos x + C2 sin x solves y'' + y = 0 for all
# C1, C2; x^2 and 1/x solve x^2 y'' - 2y = 0; e^(x^2) does not solve y' = 2y (non-zero residual).
x, C1, C2 = var("x C1 C2")
def residual(y, eq):            # eq is a function of y returning the expression that should vanish
    return eq(y).simplify_full()
y1 = C1 * cos(x) + C2 * sin(x)
print("y'' + y for C1 cos x + C2 sin x:", residual(y1, lambda y: diff(y, x, 2) + y), "-> solution:", bool(residual(y1, lambda y: diff(y, x, 2) + y) == 0))
for y2 in [x ** 2, 1 / x]:
    r = residual(y2, lambda y: x ** 2 * diff(y, x, 2) - 2 * y)
    print("x^2 y'' - 2y for y =", y2, ":", r, "-> solution:", bool(r == 0))
y3 = exp(x ** 2)
r = residual(y3, lambda y: diff(y, x) - 2 * y)
print("y' - 2y for y = e^(x^2):", r, "-> solution:", bool(r == 0))

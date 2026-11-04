# Exercise exr-l2-membership: square-integrability of 1/(1 + x^2), 1/sqrt(1 + x^2), sin(x)/x on R and of x^(-1/3),
# x^(-1/2) on (0, 1), by exact improper integrals of |f|^2.
x = var("x", domain="real")
for f in [1 / (1 + x ** 2), 1 / sqrt(1 + x ** 2), sin(x) / x]:
    print("int_R |f|^2 for f =", f, ":", integrate(f ** 2, x, -oo, oo))
for f in [x ** (-1 / 3), x ** (-1 / 2)]:
    try:
        print("int_0^1 |f|^2 for f =", f, ":", integrate(f ** 2, x, 0, 1))
    except ValueError as err:
        print("int_0^1 |f|^2 for f =", f, ": divergent (", str(err).split("\n")[0], ")")

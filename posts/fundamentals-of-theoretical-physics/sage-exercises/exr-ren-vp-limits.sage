# Exercise exr-ren-vp-limits: the renormalised vacuum polarisation Pi-hat(-Q^2) = (2 alpha/pi) int x(1-x) log(1 + x(1-x) z) dx,
# z = Q^2/m^2, for small and large z.
var('x z')
small = taylor(integrate(taylor(x * (1 - x) * log(1 + x * (1 - x) * z), z, 0, 3), x, 0, 1), z, 0, 2)
print("small z: int =", small, "  so Pi-hat = (2 alpha/pi)(", small, ")")
f = lambda zv: numerical_integral(x * (1 - x) * log(1 + x * (1 - x) * zv), 0, 1)[0]
for zv in [1e2, 1e4, 1e8]:
    print("z = %.0e: 6 int - (log z - 5/3) = %.2e" % (zv, 6 * f(zv) - (log(zv) - 5 / 3)))

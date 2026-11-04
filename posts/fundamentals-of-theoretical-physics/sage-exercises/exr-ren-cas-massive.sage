# Exercise exr-ren-cas-massive: the combination Li5(q) + x Li4(q) + x^2/3 Li3(q), q = +-e^-x, and the winding sum.
import mpmath as mpm
for x in [0.001, 1, 5]:
    for sgn in [1, -1]:
        q = sgn * mpm.e**(-x)
        c = mpm.polylog(5, q) + x * mpm.polylog(4, q) + x**2 / 3 * mpm.polylog(3, q)
        w = sum(sgn**k * mpm.e**(-k * x) * (3 + 3 * k * x + (k * x)**2) / (3 * k**5) for k in range(1, 400))
        print("m l = %-5s %-12s combination %.6f, winding sum %.6f" % (x, "periodic" if sgn > 0 else "antiperiodic", c, w))
print("zeta(5) = %.6f, 15/16 zeta(5) = %.6f" % (zeta(5).n(), (15 / 16 * zeta(5)).n()))

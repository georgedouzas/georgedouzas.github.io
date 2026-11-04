# Exercise exr-ren-dec-mom: the lepton contribution to the momentum-subtraction beta function in units of e^3/(12 pi^2).
x = var('x')
f = lambda z: 6 * numerical_integral(x^2 * (1 - x)^2 * z^2 / (1 + x * (1 - x) * z^2), 0, 1)[0]
for z in [0.1, 1, 10, 100]:
    print("mu_M/m = %-5s: %.4f" % (z, f(z)), "   z^2/5 = %.4f" % (z^2 / 5) if z < 1 else "")

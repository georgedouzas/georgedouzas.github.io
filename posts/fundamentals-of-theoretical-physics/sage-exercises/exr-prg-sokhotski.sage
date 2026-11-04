# Exercise exr-prg-sokhotski: int e^{-x^2} Im[1/(x^2 - 1 + i eps)] dx -> -pi/e as eps -> 0 (the imaginary part of the
# propagator is -pi delta(x^2 - 1)).
x = var("x"); e_const = exp(1)
for e in [1e-2, 1e-3, 1e-4]:
    f = lambda t: exp(-t ** 2) * (-e / ((t ** 2 - 1) ** 2 + e ** 2))
    val = sum(numerical_integral(f, a, b, max_points=200)[0] for a, b in [(-6, -1.5), (-1.5, -1), (-1, -0.5), (-0.5, 0.5), (0.5, 1), (1, 1.5), (1.5, 6)])
    print("eps = %.0e   integral = %.6f   -pi/e = %.6f   difference = %.2e" % (e, val, -(pi / e_const).n(), val + (pi / e_const).n()))

# Exercise exr-ren-thr-constant: (2/pi) int x(1-x) log(1 + x(1-x) z^2) dx - (2/(3 pi)) log z -> -5/(9 pi), and the matching scale e^(5/6) m.
x = var('x')
for z in [10, 1e3, 1e5]:
    d = 2 / pi * numerical_integral(x * (1 - x) * log(1 + x * (1 - x) * z^2), 0, 1)[0] - 2 / (3 * pi) * log(z)
    print("z = %.0e: %.6f" % (z, d))
print("-5/(9 pi) = %.6f, e^(5/6) = %.4f" % ((-5 / (9 * pi)).n(), exp(5 / 6).n()))

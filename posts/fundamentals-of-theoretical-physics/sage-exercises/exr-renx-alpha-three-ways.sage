# Exercise exr-renx-alpha-three-ways: 1/alpha(M_Z) with the leptons: exact one-loop (on-shell), MS-bar steps at m_l, steps at e^(5/6) m_l.
al = 1 / 137.035999; MZ = 91.1876; ml = [0.000510999, 0.1056584, 1.77686]
x = var('x')
exact = 1 / al - sum(2 / pi * numerical_integral(x * (1 - x) * log(1 + x * (1 - x) * MZ^2 / m^2), 0, 1)[0] for m in ml)
steps = 1 / al - sum(2 / (3 * pi) * log(MZ / m) for m in ml)
shifted = 1 / al - sum(2 / (3 * pi) * log(MZ / (exp(5 / 6) * m)) for m in ml)
print("(a) exact one-loop %.4f   (b) steps at m %.4f   (c) steps at e^(5/6) m %.4f" % (exact.n(), steps.n(), shifted.n()))
print("(a) - (b) = %.4f, 3 x 5/(9 pi) = %.4f" % ((exact - steps).n(), (3 * 5 / (9 * pi)).n()))

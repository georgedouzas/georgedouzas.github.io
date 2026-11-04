# Exercise exr-emx-wv-skin: skin depths delta = sqrt(2/(mu0 sigma omega)) (SI).
mu0 = 4 * pi * 1e-7; eps0 = 8.8541878128e-12
for name, s, fs in [('copper', 5.96e7, [50, 1e6, 1e9]), ('sea water', 4, [100, 1e6])]:
    for f in fs:
        w = 2 * pi * f
        print(name, f, 'Hz: delta =', n(sqrt(2 / (mu0 * s * w)), digits=3), 'm; sigma/(eps0 omega) =', n(s / (eps0 * w), digits=2))

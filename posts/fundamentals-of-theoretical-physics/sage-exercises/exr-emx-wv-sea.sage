# Exercise exr-emx-wv-sea: attenuation of the intensity over 100 m of sea water (SI).
mu0 = 4 * pi * 1e-7; s = 4
for f in [100, 1e4]:
    d = n(sqrt(2 / (mu0 * s * 2 * pi * f)))
    print(f, 'Hz: delta =', n(d, digits=3), 'm; 2z/delta =', n(200 / d, digits=3), '; intensity factor =', '%.2e' % float(exp(-200 / d)))

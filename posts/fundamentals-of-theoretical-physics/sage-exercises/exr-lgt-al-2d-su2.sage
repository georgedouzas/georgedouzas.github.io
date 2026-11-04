# Exercise exr-lgt-al-2d-su2: the exact string tension of 2D SU(2) against the leading strong-coupling form -log(beta/4).
for b in (0.25, 0.5, 1.0, 2.0, 4.0):
    u = bessel_I(2, b) / bessel_I(1, b)
    print('beta = %.2f: exact -log u = %.5f, leading -log(beta/4) = %.5f, with beta^2/24: %.5f' % (b, -log(u), -log(b / 4), -log(b / 4) + b^2 / 24))

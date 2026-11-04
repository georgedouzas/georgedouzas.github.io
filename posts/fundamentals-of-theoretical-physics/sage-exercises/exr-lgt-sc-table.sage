# Exercise exr-lgt-sc-table: the leading strong-coupling string tension -log u(beta) and its small-beta form.
rows = (('Z2', lambda b: tanh(b), lambda b: -log(b)), ('U(1)', lambda b: bessel_I(1, b) / bessel_I(0, b), lambda b: -log(b / 2)),
        ('SU(2)', lambda b: bessel_I(2, b) / bessel_I(1, b), lambda b: -log(b / 4)))
for name, u, lead in rows:
    print(name + ':', '  '.join('beta = %.2f: %.4f (%.4f)' % (b, -log(u(b)), lead(b)) for b in (0.25, 0.5, 1.0)))
print('in brackets: -log(beta), -log(beta/2), -log(beta/4)')

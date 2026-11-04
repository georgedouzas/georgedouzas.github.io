# Exercise exr-emw-pol-stokes: Stokes parameters of several polarisations.
def stokes(ex, ey):
    s0 = abs(ex)^2 + abs(ey)^2; s1 = abs(ex)^2 - abs(ey)^2
    s2 = 2 * (ex.conjugate() * ey).real(); s3 = 2 * (ex.conjugate() * ey).imag()
    return [x.simplify_full() for x in (s0, s1, s2, s3)]
cases = {'linear at 30 deg': (cos(pi/6), sin(pi/6)), 'positive helicity': (1/sqrt(2), I/sqrt(2)), 'negative helicity': (1/sqrt(2), -I/sqrt(2)), 'e_x + exp(i pi/3) e_y/2': (1, exp(I*pi/3)/2)}
for name, (ex, ey) in cases.items():
    s = stokes(SR(ex), SR(ey))
    print(name, ':', s, '; s0^2 - s1^2 - s2^2 - s3^2 =', (s[0]^2 - s[1]^2 - s[2]^2 - s[3]^2).simplify_full())

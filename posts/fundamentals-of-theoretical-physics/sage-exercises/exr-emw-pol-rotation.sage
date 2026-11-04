# Exercise exr-emw-pol-rotation: Stokes parameters in axes rotated by alpha.
a, b1, b2, c1, c2 = var('alpha b1 b2 c1 c2', domain='real')
ex = b1 + I * b2; ey = c1 + I * c2
exr = ex * cos(a) + ey * sin(a); eyr = -ex * sin(a) + ey * cos(a)
def stokes(ex, ey):
    return [(ex*ex.conjugate() + ey*ey.conjugate()).simplify_full(), (ex*ex.conjugate() - ey*ey.conjugate()).simplify_full(), (2*(ex.conjugate()*ey).real_part()).simplify_full(), (2*(ex.conjugate()*ey).imag_part()).simplify_full()]
s = stokes(ex, ey); sr = stokes(exr, eyr)
print('s0, s3 unchanged:', bool((sr[0] - s[0]).simplify_full() == 0), bool((sr[3] - s[3]).simplify_full() == 0))
print('s1\' - (s1 cos 2a + s2 sin 2a) =', (sr[1] - s[1] * cos(2 * a) - s[2] * sin(2 * a)).simplify_trig())
print('s2\' - (-s1 sin 2a + s2 cos 2a) =', (sr[2] + s[1] * sin(2 * a) - s[2] * cos(2 * a)).simplify_trig())

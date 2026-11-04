# Exercise exr-lgt-wa-u1-small-angle: the compact U(1) plaquette action expanded in the field strength.
a, e, F, beta = var('a e F beta')
th = a^2 * e * F
S = (1 / e^2) * (1 - cos(th))                 # beta = 1/e^2
s = taylor(S, a, 0, 12)
print('beta (1 - cos theta_P) =', s)
print('quadratic term a^4 F^2 / 2 (Maxwell):', bool(s.coefficient(a, 4) == F^2 / 2))
print('quartic term coefficient of a^8:', s.coefficient(a, 8), '(proportional to e^2: an irrelevant self-interaction)')

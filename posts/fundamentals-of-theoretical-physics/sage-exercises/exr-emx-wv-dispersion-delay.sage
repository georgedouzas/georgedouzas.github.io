# Exercise exr-emx-wv-dispersion-delay: dispersion delay of a pulse in a tenuous plasma.
w, wp = var('omega omega_p', domain='positive')
vg = sqrt(1 - wp^2 / w^2)
print('1/v_g to first order:', (1 / vg).series(wp, 3))
e = 1.602176634e-19; me = 9.1093837e-31; eps0 = 8.8541878128e-12; c = 299792458
nL = 1e24
delay = lambda f: e^2 * nL / (2 * eps0 * me * c * (2 * pi * f)^2)
d1, d2 = delay(400e6), delay(800e6)
print('delay at 400 MHz:', n(d1, digits=3), 's; at 800 MHz:', n(d2, digits=3), 's; difference:', n(d1 - d2, digits=3), 's')

# Exercise exr-emw-int-hydrogen-line: a classical sodium line at 589 nm (SI units).
e, eps0, c, re = 1.602e-19, 8.854e-12, 2.998e8, 2.818e-15
lam, z0, r = 589e-9, 1e-10, 1.0
w = 2 * pi.n() * c / lam; tau = 2 * re / (3 * c)
P = w^4 * (e * z0)^2 / (12 * pi.n() * eps0 * c^3)
print('w = %.3e /s, <P> = %.3e W, 1/Gamma = %.3e s, Delta lambda = %.3e m' % (w, P, 1 / (w^2 * tau), 2 * pi.n() * c * tau))
print('field amplitude at 1 m on the equator: %.3e V/m' % (e * w^2 * z0 / (4 * pi.n() * eps0 * c^2 * r)))

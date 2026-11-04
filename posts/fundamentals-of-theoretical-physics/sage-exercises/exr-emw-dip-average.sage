# Exercise exr-emw-dip-average: average power of an oscillating dipole.
th, ph, t, w, p0 = var('theta phi t w p0', domain='positive')
dP = (w^2 * p0 * cos(w * t))^2 * sin(th)^2 / (16 * pi^2)
avg = w / (2 * pi) * integrate(integrate(integrate(dP * sin(th), ph, 0, 2 * pi), th, 0, pi), t, 0, 2 * pi / w)
print('<P> =', avg.simplify_full())
e, eps0, c = 1.602e-19, 8.854e-12, 2.998e8
print('electron, z0 = 0.1 nm, w = 3e15 /s: <P> = %.2e W' % ((3e15)^4 * (e * 1e-10)^2 / (12 * pi.n() * eps0 * c^3)))

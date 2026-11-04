# Exercise exr-ems-plasma-oscillation (Macchi et al., Problem 1.4): plasma frequency omega_p^2 = n e^2/(eps0 m) in SI units.
n, e, eps0, me = 1e29, 1.602e-19, 8.854e-12, 9.109e-31
wp = sqrt(n * e^2 / (eps0 * me))
print("omega_p = %.3e s^-1, frequency %.3e Hz, wavelength 2 pi c/omega_p = %.1f nm" % (wp, wp / (2 * pi.n()), 2 * pi.n() * 2.998e8 / wp * 1e9))
t = var('t'); w = var('w', domain='positive')
print("solution of d'' = -w^2 d:", desolve(diff(function('d')(t), t, 2) == -w^2 * function('d')(t), function('d')(t), ivar=t))

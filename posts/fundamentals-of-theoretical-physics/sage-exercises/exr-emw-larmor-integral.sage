# Exercise exr-emw-larmor-integral: integrating Larmor's angular distribution.
th, ph, q, a = var('theta phi q a', domain='positive')
dP = q^2 * a^2 * sin(th)^2 / (16 * pi^2)
print('total power:', integrate(integrate(dP * sin(th), ph, 0, 2 * pi), th, 0, pi))
print('fraction within 30 degrees of the plane perpendicular to a:', integrate(sin(th)^3, th, pi/3, 2*pi/3) / integrate(sin(th)^3, th, 0, pi))

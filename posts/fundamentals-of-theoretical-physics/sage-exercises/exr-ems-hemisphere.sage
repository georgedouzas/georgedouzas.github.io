# Exercise exr-ems-hemisphere (Macchi et al., Problem 1.11): field at the centre of a uniformly charged hemispherical surface.
sigma, R, th = var('sigma R theta', domain='positive')
# ring at polar angle theta: charge 2 pi sigma R^2 sin(theta) dtheta, at height R cos(theta), radius R sin(theta)
dEz = -2 * pi * sigma * R^2 * sin(th) * R * cos(th) / (4 * pi * R^3)
Ez = integrate(dEz, th, 0, pi / 2)
print("E_z at the centre =", Ez, " (Heaviside-Lorentz; -sigma/(4 eps0) in SI)")
# direct check by integrating the Coulomb kernel over the surface
ph = var('phi')
x = R * vector([sin(th) * cos(ph), sin(th) * sin(ph), cos(th)])
Ezd = integrate(integrate(-sigma * x[2] / R^3 / (4 * pi) * R^2 * sin(th), ph, 0, 2 * pi), th, 0, pi / 2)
print("direct surface integral:", Ezd)

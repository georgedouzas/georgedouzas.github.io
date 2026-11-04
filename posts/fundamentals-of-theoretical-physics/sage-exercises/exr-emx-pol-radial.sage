# Exercise exr-emx-pol-radial: bound charges and field of P = k x inside a ball of radius R (Heaviside-Lorentz).
x, y, z, k, R, r = var('x y z k R r')
assume(R > 0, r > 0)
P = vector([k * x, k * y, k * z])
rho_b = -sum(diff(P[i], v) for i, v in enumerate([x, y, z]))
print('rho_b =', rho_b)
sigma_b = k * R   # P . n at r = R
print('sigma_b =', sigma_b)
Qv = rho_b * 4 / 3 * pi * R^3
Qs = sigma_b * 4 * pi * R^2
print('total volume charge:', Qv, '; total surface charge:', Qs, '; sum:', Qv + Qs)
Er_in = rho_b * 4 / 3 * pi * r^3 / (4 * pi * r^2)
print('E_r inside (Gauss):', Er_in.simplify_full(), '; E outside: 0 since the total charge vanishes')

# Exercise exr-ems-energy-ball (Macchi et al., Problem 1.3): energy of a uniformly charged ball in three ways.
Q, R, r = var('Q R r', domain='positive')
rho = 3 * Q / (4 * pi * R^3)
# (a) shells: charge q(r) = Q r^3/R^3 at potential q/(4 pi r), add dq = rho 4 pi r^2 dr
Wa = integrate(Q * r^3 / R^3 / (4 * pi * r) * rho * 4 * pi * r^2, r, 0, R)
# (b) field energy
Win = integrate((Q * r / (4 * pi * R^3))^2 / 2 * 4 * pi * r^2, r, 0, R); Wout = integrate((Q / (4 * pi * r^2))^2 / 2 * 4 * pi * r^2, r, R, oo)
# (c) (1/2) int rho phi
phi_in = Q * (3 * R^2 - r^2) / (8 * pi * R^3)
Wc = integrate(rho * phi_in / 2 * 4 * pi * r^2, r, 0, R)
print("(a)", Wa.simplify_full(), " (b)", (Win + Wout).simplify_full(), " (c)", Wc.simplify_full())
print("fraction of the field energy outside:", (Wout / (Win + Wout)).simplify_full())

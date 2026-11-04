# Exercise exr-emm-shell-angular-momentum: angular momentum of the field outside a rotating charged shell.
r, th, Q, a, w = var('r theta Q a w', domain='positive')
m = Q * a^2 * w / 3
Sphi = Q * m * sin(th) / (16 * pi^2 * r^5)         # E_r B_theta with E_r = Q/(4 pi r^2), B_theta = m sin/(4 pi r^3)
Jz = integrate(integrate(r * sin(th) * Sphi * 2 * pi * r^2 * sin(th), th, 0, pi), r, a, oo)
print('J_z =', Jz.simplify_full(), '; Q m/(6 pi a) =', (Q * m / (6 * pi * a)).simplify_full())

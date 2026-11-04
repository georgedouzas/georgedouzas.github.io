# Exercise exr-emx-wg-power: time-averaged power and energy per unit length of the TE10 mode.
x, y, z, t, k, w, E0, a, b = var('x y z t k omega E0 a b', domain='positive')
Ey = E0 * sin(pi * x / a) * cos(k * z - w * t)
Bx = -k / w * Ey
Bz = pi * E0 / (w * a) * cos(pi * x / a) * sin(k * z - w * t)
T = 2 * pi / w
avg = lambda f: integrate(integrate(integrate(f, x, 0, a), y, 0, b), t, 0, T) / T
P = avg(-Ey * Bx).simplify_full()
U = avg((Ey^2 + Bx^2 + Bz^2) / 2).simplify_full()
print('<P> =', P)
print('<U> per unit length =', U.subs({w: sqrt(k^2 + pi^2 / a^2)}).simplify_full())
print('ratio =', (P / U).subs({w: sqrt(k^2 + pi^2 / a^2)}).simplify_full())

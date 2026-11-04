# Exercise exr-emx-wg-check: the TE10 mode of a rectangular guide 0 < x < a (Heaviside-Lorentz, c = 1).
x, y, z, t, k, w, E0, a = var('x y z t k omega E0 a')
assume(a > 0)
E = vector([0, E0 * sin(pi * x / a) * cos(k * z - w * t), 0])
B = vector([-k / w * E[1], 0, pi * E0 / (w * a) * cos(pi * x / a) * sin(k * z - w * t)])
X = [x, y, z]
curl = lambda F: vector([diff(F[2], y) - diff(F[1], z), diff(F[0], z) - diff(F[2], x), diff(F[1], x) - diff(F[0], y)])
div = lambda F: sum(diff(F[i], X[i]) for i in range(3))
print('Faraday curl E + dB/dt:', (curl(E) + diff(B, t)).simplify_full())
print('div E, div B:', div(E).simplify_full(), div(B).simplify_full())
res = (curl(B) - diff(E, t)).simplify_full()
print('Ampere-Maxwell residual:', vector([c.factor() for c in res]))
print('vanishes when omega^2 = k^2 + pi^2/a^2:', bool(res.subs({w: sqrt(k^2 + pi^2 / a^2)}).simplify_full() == 0))
print('E_y on x = 0, a:', E[1].subs(x=0), E[1].subs(x=a), '; B_x on x = 0, a:', B[0].subs(x=0), B[0].subs(x=a))

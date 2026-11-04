# Exercise exr-emm-standing-wave: a standing wave, Maxwell's equations and Poynting's theorem.
t, x, y, z = var("t x y z")
X = [x, y, z]
def div(F): return sum(diff(F[i], X[i]) for i in range(3)).simplify_full()
def curl(F): return vector([diff(F[2], y) - diff(F[1], z), diff(F[0], z) - diff(F[2], x), diff(F[1], x) - diff(F[0], y)]).simplify_full()
def grad(f): return vector([diff(f, v) for v in X])
E0 = var('E0')
E = vector([E0 * sin(z) * cos(t), 0, 0]); B = vector([0, -E0 * cos(z) * sin(t), 0])
print('div E, div B:', div(E), div(B), '; curl E + dB/dt =', (curl(E) + diff(B, t)).simplify_full(), '; curl B - dE/dt =', (curl(B) - diff(E, t)).simplify_full())
u = (E.dot_product(E) + B.dot_product(B)) / 2; S = E.cross_product(B)
print('S =', S, '; du/dt + div S =', (diff(u, t) + div(S)).simplify_full())
print('time average of S_z:', integrate(S[2], t, 0, 2 * pi) / (2 * pi))

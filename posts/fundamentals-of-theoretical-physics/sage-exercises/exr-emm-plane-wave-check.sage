# Exercise exr-emm-plane-wave-check: a plane wave solves the vacuum Maxwell equations; a wrongly oriented B does not.
t, x, y, z = var("t x y z")
X = [x, y, z]
def div(F): return sum(diff(F[i], X[i]) for i in range(3)).simplify_full()
def curl(F): return vector([diff(F[2], y) - diff(F[1], z), diff(F[0], z) - diff(F[2], x), diff(F[1], x) - diff(F[0], y)]).simplify_full()
def grad(f): return vector([diff(f, v) for v in X])
E0 = var('E0')
E = vector([E0 * cos(z - t), 0, 0])
for name, B in [('B along e_y', vector([0, E0 * cos(z - t), 0])), ('B along e_x', vector([E0 * cos(z - t), 0, 0]))]:
    print(name, ': div E =', div(E), ', div B =', div(B), ', curl E + dB/dt =', (curl(E) + diff(B, t)).simplify_full(), ', curl B - dE/dt =', (curl(B) - diff(E, t)).simplify_full())

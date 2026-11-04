# Exercise exr-emm-sym-galilean-check: a Galilean boost of a plane wave is not a solution.
t, x, y, z = var("t x y z")
X = [x, y, z]
def div(F): return sum(diff(F[i], X[i]) for i in range(3)).simplify_full()
def curl(F): return vector([diff(F[2], y) - diff(F[1], z), diff(F[0], z) - diff(F[2], x), diff(F[1], x) - diff(F[0], y)]).simplify_full()
def grad(f): return vector([diff(f, v) for v in X])
def maxwell(E, B): return [div(E), div(B), (curl(E) + diff(B, t)).simplify_full(), (curl(B) - diff(E, t)).simplify_full()]
E0, v = var('E0 v')
# fields written in the primed coordinates (z = z' + v t'), with the primed variables named z, t
E = vector([E0 * cos(z + v * t - t), 0, 0]); B = vector([0, E0 * cos(z + v * t - t), 0])
m = maxwell(E, B)
print('div E, div B:', m[0], m[1])
print('curl E + dB/dt =', m[2].apply_map(lambda e: e.reduce_trig()), '; curl B - dE/dt =', m[3].apply_map(lambda e: e.reduce_trig()))

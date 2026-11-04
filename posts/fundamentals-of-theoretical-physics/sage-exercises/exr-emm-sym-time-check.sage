# Exercise exr-emm-sym-time-check: time reversal applied to a plane wave.
t, x, y, z = var("t x y z")
X = [x, y, z]
def div(F): return sum(diff(F[i], X[i]) for i in range(3)).simplify_full()
def curl(F): return vector([diff(F[2], y) - diff(F[1], z), diff(F[0], z) - diff(F[2], x), diff(F[1], x) - diff(F[0], y)]).simplify_full()
def grad(f): return vector([diff(f, v) for v in X])
def maxwell(E, B): return [div(E), div(B), (curl(E) + diff(B, t)).simplify_full(), (curl(B) - diff(E, t)).simplify_full()]
E0 = var('E0')
E = vector([E0 * cos(z - t), 0, 0]); B = vector([0, E0 * cos(z - t), 0])
Et = E.subs(t=-t); Bt = -B.subs(t=-t)
print('transformed fields:', Et, Bt); print('vacuum equations:', maxwell(Et, Bt))
print('Poynting vector before:', E.cross_product(B).simplify_full(), '; after:', Et.cross_product(Bt).simplify_full())

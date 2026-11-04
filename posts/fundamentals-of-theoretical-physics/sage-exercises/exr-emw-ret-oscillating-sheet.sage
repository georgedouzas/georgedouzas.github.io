# Exercise exr-emw-ret-oscillating-sheet: a current sheet K0 cos(w t) e_x on z = 0, field for z > 0.
t, x, y, z = var("t x y z")
X = [x, y, z]
def div(F): return sum(diff(F[i], X[i]) for i in range(3)).simplify_full()
def curl(F): return vector([diff(F[2], y) - diff(F[1], z), diff(F[0], z) - diff(F[2], x), diff(F[1], x) - diff(F[0], y)]).simplify_full()
def grad(f): return vector([diff(f, v) for v in X])
def maxwell(E, B): return [div(E), div(B), (curl(E) + diff(B, t)).simplify_full(), (curl(B) - diff(E, t)).simplify_full()]
box = lambda f: (diff(f, t, 2) - sum(diff(f, v, 2) for v in X)).simplify_full()
K0, w = var('K0 w', domain='positive')
A = vector([K0 / (2 * w) * sin(w * (t - z)), 0, 0])
E = -diff(A, t); B = curl(A)
print('E =', E, '; B =', B, '; vacuum equations:', [m.simplify_trig() if hasattr(m, 'simplify_trig') else m.apply_map(lambda e: e.simplify_trig()) for m in maxwell(E, B)])
Sz = (E.cross_product(B))[2]
print('average Poynting flux on one side:', (w / (2 * pi) * integrate(Sz, t, 0, 2 * pi / w)).simplify_full())

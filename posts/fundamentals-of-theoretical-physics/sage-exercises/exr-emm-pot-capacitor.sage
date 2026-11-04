# Exercise exr-emm-pot-capacitor: potentials of a capacitor beyond the quasi-static approximation.
t, x, y, z = var("t x y z")
X = [x, y, z]
def curl(F): return vector([diff(F[2], y) - diff(F[1], z), diff(F[0], z) - diff(F[2], x), diff(F[1], x) - diff(F[0], y)]).simplify_full()
def grad(f): return vector([diff(f, v) for v in X])
Ef = function('E')(t); w, a = var('w a', domain='positive')
for name, Et in [('general E(t)', Ef), ('E = cos(w t)', cos(w * t))]:
    phi = -Et * z; A = vector([0, 0, -diff(Et, t) * (x^2 + y^2) / 4])
    Efield = (-grad(phi) - diff(A, t)).simplify_full(); B = curl(A)
    print(name, ': E =', Efield, '; B =', B)
print('relative correction at the rim for E = cos(w t):', ((-grad(-cos(w*t)*z) - diff(vector([0, 0, w*sin(w*t)*(x^2 + y^2)/4]), t))[2] / cos(w*t) - 1).subs(x=a, y=0).simplify_full())

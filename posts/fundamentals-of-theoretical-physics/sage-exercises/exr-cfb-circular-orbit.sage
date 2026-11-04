# Exercise exr-cfb-circular-orbit (Maggiore, Problem 10.2): a charge on a circle of radius a in the xy-plane with
# frequency w (a w << c). The dipole d = q a (cos w t, sin w t, 0); time-averaged angular distribution proportional to
# (1 + cos^2 t)/2, total power q^2 a^2 w^4/(6 pi eps0 c^3), and the polarisation seen along z and in the plane.
th, ph, t, q, a, w, e0, c = var('th ph t q a w e0 c'); assume(q > 0, a > 0, w > 0, e0 > 0, c > 0)
dd = -q * a * w^2 * vector([cos(w * t), sin(w * t), 0])
n = vector([sin(th) * cos(ph), sin(th) * sin(ph), cos(th)])
perp2 = (dd * dd - (n * dd)^2)                          # |n x ddot d|^2
dP = perp2 / (16 * pi^2 * e0 * c^3)
avg = (w / (2 * pi) * integrate(dP, t, 0, 2 * pi / w)).simplify_full()
print("time-averaged dP/dOmega =", avg.factor())
print("proportional to (1 + cos^2 th)/2:", bool((avg / ((1 + cos(th)^2) / 2) - q^2 * a^2 * w^4 / (16 * pi^2 * e0 * c^3)).simplify_full() == 0))
P = integrate(integrate(avg * sin(th), th, 0, pi), ph, 0, 2 * pi).simplify_full()
print("total power:", P, " = q^2 a^2 w^4/(6 pi eps0 c^3):", bool((P - q^2 * a^2 * w^4 / (6 * pi * e0 * c^3)).simplify_full() == 0))
Ez = (dd - (n * dd) * n).subs(th=0)                     # transverse part seen along z: rotating, circular polarisation
print("field seen along z (transverse part of ddot d):", [e.simplify_full() for e in Ez])
Ex = (dd - (n * dd) * n).subs(th=pi / 2, ph=pi / 2)     # seen along y in the plane: only the x component, linear
print("field seen along y in the plane:", [e.simplify_full() for e in Ex])

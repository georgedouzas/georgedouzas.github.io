# Exercise exr-emw-dip-rotating: angular pattern of a rotating dipole.
th, ph, t, w, p0 = var('theta phi t w p0', domain='positive')
n = vector([sin(th) * cos(ph), sin(th) * sin(ph), cos(th)])
pdd = -w^2 * p0 * vector([cos(w * t), sin(w * t), 0])
f = n.cross_product(pdd); f2 = f.dot_product(f)
avg = (w / (2 * pi) * integrate(f2.expand(), t, 0, 2 * pi / w)).simplify_full()
print('time average of |n x p..|^2:', avg.simplify_trig())
P = integrate(integrate(avg / (16 * pi^2) * sin(th), ph, 0, 2 * pi), th, 0, pi)
print('total power:', P.simplify_full())

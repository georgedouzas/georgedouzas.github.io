# Exercise exr-emm-pot-solenoid-ac: the electric field outside a solenoid with alternating current.
t, x, y, z = var("t x y z")
X = [x, y, z]
def div(F): return sum(diff(F[i], X[i]) for i in range(3)).simplify_full()
def curl(F): return vector([diff(F[2], y) - diff(F[1], z), diff(F[0], z) - diff(F[2], x), diff(F[1], x) - diff(F[0], y)]).simplify_full()
def grad(f): return vector([diff(f, v) for v in X])
n, I0, w, a, s, ph = var('n I0 w a s ph'); assume(a > 0, s > a)
Aout = n * I0 * sin(w * t) * a^2 / (2 * (x^2 + y^2)) * vector([-y, x, 0])
E = -diff(Aout, t)
print('E =', E.simplify_full(), '; curl E =', curl(E), '; B = curl A =', curl(Aout))
circ = integrate((E[0].subs(x=s*cos(ph), y=s*sin(ph)) * (-s*sin(ph)) + E[1].subs(x=s*cos(ph), y=s*sin(ph)) * s*cos(ph)).simplify_full(), ph, 0, 2*pi)
print('circulation:', circ.simplify_full(), '; -dPhi/dt =', -diff(pi * a^2 * n * I0 * sin(w * t), t))

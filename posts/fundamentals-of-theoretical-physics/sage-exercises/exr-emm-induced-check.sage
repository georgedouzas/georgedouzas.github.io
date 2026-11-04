# Exercise exr-emm-induced-check (Pierrus, Question 5.12): the field induced by a growing uniform field in a cylinder.
t, x, y, z = var("t x y z")
X = [x, y, z]
def div(F): return sum(diff(F[i], X[i]) for i in range(3)).simplify_full()
def curl(F): return vector([diff(F[2], y) - diff(F[1], z), diff(F[0], z) - diff(F[2], x), diff(F[1], x) - diff(F[0], y)]).simplify_full()
def grad(f): return vector([diff(f, v) for v in X])
B = function('B')(t); r0 = var('r0')
Ein = diff(B, t) / 2 * vector([y, -x, 0]); Eout = diff(B, t) * r0^2 / (2 * (x^2 + y^2)) * vector([y, -x, 0])
print('inside: curl E =', curl(Ein), '; -dB/dt e_z =', (0, 0, -diff(B, t)), '; div E =', div(Ein))
print('outside: curl E =', curl(Eout), '; div E =', div(Eout))
s, ph = var('s ph'); assume(s > 0)
circ = integrate((Eout[0].subs(x=s*cos(ph), y=s*sin(ph)) * (-s*sin(ph)) + Eout[1].subs(x=s*cos(ph), y=s*sin(ph)) * s*cos(ph)).simplify_full(), ph, 0, 2*pi)
print('circulation outside at radius s:', circ, '= -pi r0^2 dB/dt')

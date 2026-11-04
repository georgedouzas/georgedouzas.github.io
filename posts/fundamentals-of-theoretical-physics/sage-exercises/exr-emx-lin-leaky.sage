# Exercise exr-emx-lin-leaky: leaky spherical capacitor.
r, a, b, eps, sig, Q0 = var('r a b epsilon sigma Q0', domain='positive')
t = var('t'); assume(b > a)
V = integrate(1 / (4 * pi * eps * r^2), r, a, b)     # potential difference per unit charge
C = (1 / V).simplify_full()
R = integrate(1 / (4 * pi * sig * r^2), r, a, b).simplify_full()
print('C =', C.factor(), '; R =', R.factor(), '; RC =', (R * C).simplify_full())
Q = function('Q')(t)
sol = desolve(R * diff(Q, t) + Q / C, Q, ics=[0, Q0], ivar=t)
print('Q(t) =', sol.simplify_full())

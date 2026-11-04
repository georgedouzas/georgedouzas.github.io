# Exercise exr-cfb-dipole-power (Maggiore, Problem 10.1): an electric dipole d(t) = q l cos(w t) along z (l w << c).
# Angular distribution dP/dOmega = ddot(d)^2 sin^2(t)/(16 pi^2 eps0 c^3), total power and its time average, and the
# numbers for an electron with l = 1e-10 m at w = 3e15 s^-1.
th, ph, q, l, w, e0, c, t = var('th ph q l w e0 c t'); assume(q > 0, l > 0, w > 0, e0 > 0, c > 0)
dd = -q * l * w^2 * cos(w * t)
dP = dd^2 * sin(th)^2 / (16 * pi^2 * e0 * c^3)
P = integrate(integrate(dP * sin(th), th, 0, pi), ph, 0, 2 * pi).simplify_full()
print("P(t) =", P, "  = ddot(d)^2/(6 pi eps0 c^3):", bool((P - dd^2 / (6 * pi * e0 * c^3)).simplify_full() == 0))
Pav = (w / (2 * pi) * integrate(P, t, 0, 2 * pi / w)).simplify_full()
print("time average:", Pav)
num = {q: 1.602e-19, l: 1e-10, w: 3e15, e0: 8.854e-12, c: 2.998e8}
print("numbers: <P> = %.3e W, energy hbar w = %.3e J, photons per second ~ %.2e" % (Pav.subs(num), 1.0546e-34 * 3e15, (Pav.subs(num) / (1.0546e-34 * 3e15))))
print("fraction of the power within 30 deg of the equator:", (integrate(sin(th)^3, th, pi / 3, 2 * pi / 3) / integrate(sin(th)^3, th, 0, pi)).n(digits=4))

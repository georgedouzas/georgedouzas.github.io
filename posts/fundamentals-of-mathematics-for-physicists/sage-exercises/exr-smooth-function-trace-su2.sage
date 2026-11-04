# Exercise exr-smooth-function-trace-su2: the trace of the Euler-angle matrix of SU(2) is 2 cos(theta/2) cos((phi+psi)/2),
# a smooth function with values in [-2, 2]; it equals 2 only at the identity (theta = 0, phi + psi = 0 mod 4 pi) and -2
# only at -1 (theta = 0, phi + psi = 2 pi mod 4 pi), both outside the Euler chart.
th, ph, ps = var("th ph ps", domain="real")
a = cos(th/2)*exp(-I*(ph + ps)/2); b = -I*sin(th/2)*exp(I*(ph - ps)/2)
g = matrix(SR, [[a, -conjugate(b)], [b, conjugate(a)]])
tr = g.trace().rectform().simplify_full()
print("tr g =", tr)
print("equals 2 cos(theta/2) cos((phi + psi)/2):", bool((tr - 2*cos(th/2)*cos((ph + ps)/2)).simplify_full() == 0))
print("equals a + conj(a) = 2 Re a:", bool((tr - 2*a.real_part()).simplify_full() == 0))
print("at theta = 0, phi + psi = 0 the matrix is the identity:", g.subs(th=0, ps=-ph).apply_map(lambda e: e.rectform().simplify_full()))
print("at theta = 0, phi + psi = 2 pi the matrix is -1:", g.subs(th=0, ps=2*pi - ph).apply_map(lambda e: e.rectform().simplify_full()))
print("on the chart 0 < theta < pi, cos(theta/2) lies strictly between 0 and 1, so |tr g| < 2 there")
val = tr.subs(th=pi/3, ph=pi/4, ps=pi/2).simplify_full()
print("sample at (theta, phi, psi) = (pi/3, pi/4, pi/2): tr g =", val, "=", N(val, digits=6))

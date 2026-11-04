# Exercise exr-real-integral-trig: the integral of 1/(2 + cos theta) over [0, 2 pi] by the substitution z = e^{i theta}
# and the residue at the root of z^2 + 4 z + 1 inside the unit circle, and by Sage's integrate.
th = var("theta", domain="real"); z = var("z")
direct = integrate(1/(2 + cos(th)), th, 0, 2*pi)
print("integrate:", direct)
g = 2/(I*(z^2 + 4*z + 1))                   # d theta = dz/(i z), cos theta = (z + 1/z)/2
roots = [r for r, m in (z^2 + 4*z + 1).roots(ring=QQbar)]
inside = [r for r in roots if abs(r) < 1]
print("roots:", roots, " inside the unit circle:", inside)
res = g.residue(z == -2 + sqrt(3))
print("2 pi i * residue =", (2*pi*I*res).simplify_full(), "  equals 2 pi/sqrt(3):", bool(((2*pi*I*res) - 2*pi/sqrt(3)).simplify_full() == 0))

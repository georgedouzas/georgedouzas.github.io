# Exercise exr-circle-rational: the circle integral of 3 z^2 + 2/z + 1/z^2 + 5/z^3 over |z| = 1 is 2 pi i times the
# coefficient of 1/z, by linearity of the integral and the values of the circle integrals of z^n.
t = var("t", domain="real"); z = var("z")
f = 3*z^2 + 2/z + 1/z^2 + 5/z^3
g = exp(I*t)
val = integrate(f.subs(z=g)*I*g, t, 0, 2*pi)
print("parametrised integral:", val)
print("2 pi i * residue at 0:", 2*pi*I*f.residue(z == 0))
print("equal to 4 pi i:", bool(val == 4*pi*I))

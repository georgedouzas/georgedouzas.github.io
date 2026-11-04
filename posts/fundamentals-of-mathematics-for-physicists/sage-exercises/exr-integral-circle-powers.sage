# Exercise exr-integral-circle-powers: circle integrals of z^n for n = -2..2 and of conj(z) over |z| = 1 by the
# parametrisation z = e^{it}, dz = i e^{it} dt.
t = var("t", domain="real")
g = exp(I*t); gp = I*exp(I*t)
for n in range(-2, 3):
    print("n = %2d: integral of z^n dz = %s" % (n, integrate(g^n*gp, t, 0, 2*pi)))
print("integral of conj(z) dz =", integrate(conjugate(g)*gp, t, 0, 2*pi), " (conj(z) = 1/z on the circle)")

# Exercise exr-cif-cos: the integral of cos z/(z - pi) over |z| = 4 by Cauchy's integral formula (as a residue) and
# numerically; over |z| = 1 the integrand is holomorphic and the integral is 0.
z = var("z"); t = var("t", domain="real")
f = cos(z)/(z - pi)
print("residue at pi:", f.residue(z == pi), " so the integral over |z| = 4 is", 2*pi*I*f.residue(z == pi))
def contour(R):
    g = R*exp(I*t); h = f.subs(z=g)*I*g
    return CC(numerical_integral(lambda s: h.subs(t=s).real(), 0, 2*pi)[0], numerical_integral(lambda s: h.subs(t=s).imag(), 0, 2*pi)[0])
print("numerical, |z| = 4:", contour(4))
print("numerical, |z| = 1:", contour(1))

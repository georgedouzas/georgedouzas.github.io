# Exercise exr-residue-theorem-sum: the integral of (z^2 + 1)/(z (z - 1)^2) over |z| = 3 by the residue theorem and
# by numerical integration of the parametrised contour.
z = var("z"); t = var("t", domain="real")
f = (z^2 + 1)/(z*(z - 1)^2)
total = 2*pi*I*(f.residue(z == 0) + f.residue(z == 1))
print("2 pi i (Res_0 + Res_1) =", total)
g = 3*exp(I*t); h = f.subs(z=g)*I*g
num = CC(numerical_integral(lambda s: h.subs(t=s).real(), 0, 2*pi)[0], numerical_integral(lambda s: h.subs(t=s).imag(), 0, 2*pi)[0])
print("numerical contour integral:", num)
print("agree to 1e-8:", abs(num - total.n()) < 1e-8)

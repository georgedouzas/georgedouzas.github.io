# Exercise exr-cif-derivative: the derivative form of Cauchy's integral formula for e^{2z}/z^3 over |z| = 1 and
# z^2/(z - 1)^2 over |z| = 2, as residues and as derivatives.
z = var("z")
f1 = exp(2*z)/z^3
print("integral of e^{2z}/z^3 over |z| = 1: 2 pi i * Res =", 2*pi*I*f1.residue(z == 0), "  formula 2 pi i/2! * (e^{2z})''(0) =", 2*pi*I/2*diff(exp(2*z), z, 2).subs(z=0))
f2 = z^2/(z - 1)^2
print("integral of z^2/(z - 1)^2 over |z| = 2: 2 pi i * Res =", 2*pi*I*f2.residue(z == 1), "  formula 2 pi i * (z^2)'(1) =", 2*pi*I*diff(z^2, z).subs(z=1))

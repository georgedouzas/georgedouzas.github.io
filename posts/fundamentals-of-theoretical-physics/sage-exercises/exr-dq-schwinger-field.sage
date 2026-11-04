# Exercise exr-dq-schwinger-field: the critical field m^2 c^3/(e hbar) and the proton's field at the Bohr radius; their ratio is 1/alpha^3.
me = 9.1093837015e-31; c = 299792458; e = 1.602176634e-19; hbar = 1.054571817e-34; eps0 = 8.8541878128e-12; alpha = 1 / 137.035999166
Ec = me^2 * c^3 / (e * hbar); a0 = hbar / (me * c * alpha)
Eb = e / (4 * pi * eps0 * a0^2)
print("E_c = %.3e V/m; a_0 = %.4e m; field at a_0 = %.3e V/m; ratio = %.4e; 1/alpha^3 = %.4e" % (Ec, a0, Eb.n(), (Ec / Eb).n(), alpha^-3))

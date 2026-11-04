# Exercise exr-torsion-sage: the connection Gamma^x_xy = 1 (all other coefficients zero) on the plane has torsion
# T^x_xy = 1 and vanishing curvature: flat, but not torsion-free (SageManifolds affine connection). SageManifolds
# stores nabla_{e_j} e_i = Gamma^k_ij e_k, derivative index last, so the book's Gamma^x_xy is nab[0, 1, 0].
from sage.all import *
M = Manifold(2, 'R^2'); X = M.chart('x y')
nab = M.affine_connection('nabla', latex_name=r'\nabla')
nab[0, 1, 0] = 1     # book's Gamma^x_xy = 1 (derivative index first); SageManifolds puts the derivative index last
print("coefficients:", nab.display())
T = nab.torsion()
print("torsion:", T.display())
print("T^x_xy = 1 and T^x_yx = -1:", bool(T[0, 0, 1].expr() == 1 and T[0, 1, 0].expr() == -1))
print("Riemann tensor vanishes (flat):", nab.riemann() == 0)
sym = M.affine_connection('nablaS')                          # the symmetric part Gamma^k_(ij)
for k in range(2):
    for i in range(2):
        for j in range(2):
            sym[k, i, j] = (nab[k, i, j].expr() + nab[k, j, i].expr())/2
print("symmetric part Gamma^x_xy = Gamma^x_yx = 1/2:", bool(sym[0, 0, 1].expr() == 1/2 and sym[0, 1, 0].expr() == 1/2), "; its torsion vanishes:", sym.torsion() == 0)
print("but the symmetric part is curved: R^x_yxy =", sym.riemann()[0, 1, 0, 1].expr())

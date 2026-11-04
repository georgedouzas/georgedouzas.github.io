# Exercise exr-qmn-finite-well-matrix (Izaac-Wang, Sec. 9.6 and Problem P9.7): the direct matrix method for the finite
# square well V = -V0 on |x| < a, in units hbar^2/2m = 1, a = 1, V0 = z0^2 with z0 = 3. The grid covers [-X, X] with the
# walls psi(-X) = psi(X) = 0; the lowest eigenvalues of the tridiagonal matrix are compared with the roots of the even
# and odd matching conditions of exr-qmb-finite-well, (E + V0)/V0 = 0.1521 and 0.5770.
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq
z0 = 3.0; V0 = z0^2
ze = brentq(lambda z: z*np.tan(z) - np.sqrt(z0^2 - z^2), 1e-9, np.pi/2 - 1e-9)
zo = brentq(lambda z: -z/np.tan(z) - np.sqrt(z0^2 - z^2), np.pi/2 + 1e-9, z0 - 1e-12)
exact = [float((ze/z0)^2*V0 - V0), float((zo/z0)^2*V0 - V0)]
print("exact energies E = %.6f, %.6f" % tuple(exact))
def levels(X, dx):
    x = np.arange(-X + dx, X - dx/2, dx)
    V = np.where(np.abs(x) < 1, -V0, 0.0)
    E = eigh_tridiagonal(2/dx^2 + V, -np.ones(len(x) - 1)/dx^2, select='i', select_range=(0, 2))[0]
    return E
for dx in (0.1, 0.05, 0.025, 0.0125):
    E = levels(6.0, dx)
    print("dx = %.4f: E0 = %.6f (error %.1e), E1 = %.6f (error %.1e), next eigenvalue %.4f > 0" % (dx, E[0], E[0] - exact[0], E[1], E[1] - exact[1], E[2]))
print("box size: X = 3, 6, 9 at dx = 0.0125 ->", [round(levels(X, 0.0125)[0], 6) for X in (3.0, 6.0, 9.0)])

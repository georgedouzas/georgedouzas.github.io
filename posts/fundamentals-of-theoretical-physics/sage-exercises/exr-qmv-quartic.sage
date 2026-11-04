# Exercise exr-qmv-quartic (Izaac-Wang, Sec. 10.3, the variational principle; the matrix check as in Problem P9.7):
# H = p^2/2 + x^4 (units hbar = m = 1). The Gaussian trial psi = exp(-x^2/4 s^2), whose density has variance s^2, has
# <p^2> = 1/(4 s^2) and <x^4> = 3 s^4; the bound E(s) = 1/(8 s^2) + 3 s^4 is minimised and compared with the ground
# state of the finite-difference matrix on [-6, 6].
import numpy as np
from scipy.linalg import eigh_tridiagonal
s, x = var('s x'); assume(s > 0)
psi = exp(-x^2/(4*s^2))
nrm = integrate(psi^2, x, -oo, oo)
E = (integrate(diff(psi, x)^2, x, -oo, oo)/2 + integrate(x^4*psi^2, x, -oo, oo))/nrm
print("E(s) =", E.simplify_full())
s0 = 48^(-1/6)
assert diff(E, s).subs(s=s0).simplify_full() == 0
print("minimum at s^6 = 1/48, s = %.6f, E = %s = %.6f" % (s0.n(), E.subs(s=s0).simplify_full(), E.subs(s=s0).n()))
dx = 0.005; xs = np.arange(-6 + dx, 6 - dx/2, dx)
E0 = eigh_tridiagonal(1/dx^2 + xs^4, -0.5/dx^2*np.ones(len(xs) - 1), select='i', select_range=(0, 0))[0][0]
print("grid ground state E0 = %.6f; the Gaussian bound lies above it by %.2f%%" % (E0, 100*(E.subs(s=s0).n()/E0 - 1)))

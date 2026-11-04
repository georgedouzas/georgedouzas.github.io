# Exercise exr-qmn-morse (Izaac-Wang, Problems P9.4, P9.7 and P9.8): the Morse potential
# V = C0 + De (1 - exp(-a (x - xe)))^2 with C0 = 6.4, De = 0.3, a = 1, xe = 1.6 in units hbar = m = 1. Its exact ground
# state energy is E0 = C0 + w/2 - w^2/(16 De) with w = a sqrt(2 De) = sqrt(0.6), i.e. (251 + 4 sqrt 15)/40, and it is
# the only bound state (the levels n < sqrt(2 De)/a - 1/2 = 0.27 are bound). Second-order finite differences
# (H = -A/2 + V) and the Numerov matrix (H = -B^{-1} A/2 + V, A = tridiag(1, -2, 1)/dx^2, B = tridiag(1, 10, 1)/12) are
# compared on the grid [-2, 80] (the state is bound by only 0.038, so its tail exp(-0.275 x) is long) with psi = 0 at both ends.
import numpy as np
E0 = (251 + 4*sqrt(15))/40
print("exact E0 = (251 + 4 sqrt 15)/40 = %.8f, dissociation threshold C0 + De = 6.7" % E0.n())
def H_fd(x, dx, V):
    n = len(x)
    A = (np.diag(-2*np.ones(n)) + np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1))/dx^2
    B = (np.diag(10*np.ones(n)) + np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1))/12
    return -A/2 + np.diag(V), -np.linalg.solve(B, A)/2 + np.diag(V)
for dx in (0.4, 0.2, 0.1):
    x = np.arange(-2 + dx, 80 - dx/2, dx)
    V = 6.4 + 0.3*(1 - np.exp(-(x - 1.6)))^2
    Hf, Hn = H_fd(x, dx, V)
    ef = np.sort(np.linalg.eigvalsh(Hf))[0]
    en = np.sort(np.linalg.eigvals(Hn).real)[0]
    print("dx = %.2f: finite differences %.8f (error %.1e), Numerov %.8f (error %.1e)" % (dx, ef, ef - E0.n(), en, en - E0.n()))
x = np.arange(-2 + 0.1, 80 - 0.05, 0.1); V = 6.4 + 0.3*(1 - np.exp(-(x - 1.6)))^2
ev = np.sort(np.linalg.eigvalsh(-((np.diag(-2*np.ones(len(x))) + np.diag(np.ones(len(x) - 1), 1) + np.diag(np.ones(len(x) - 1), -1))/0.1^2)/2 + np.diag(V)))
print("eigenvalues below the threshold 6.7:", [round(e, 5) for e in ev if e < 6.7], "; next:", round(ev[1], 5))

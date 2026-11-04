# Exercise exr-clm-int-kepler: Kepler problem (m = k = 1) integrated from (x, y, px, py) = (1, 0, 0, 0.8); energy,
# angular momentum and the Laplace-Runge-Lenz vector stay constant, and the orbit is the conic with e = |A|.
import numpy as np
from scipy.integrate import solve_ivp
def rhs(t, s):
    r3 = (s[0]**2 + s[1]**2)**1.5
    return [s[2], s[3], -s[0] / r3, -s[1] / r3]
s0 = [1.0, 0.0, 0.0, 0.8]
sol = solve_ivp(rhs, [0, 20], s0, rtol=1e-11, atol=1e-11, dense_output=True)
ts = np.linspace(0, 20, 2001); Y = sol.sol(ts)
x, y, px, py = Y; r = np.sqrt(x**2 + y**2)
E = (px**2 + py**2) / 2 - 1 / r; L = x * py - y * px
Ax = py * L - x / r; Ay = -px * L - y / r
print("E: initial %.6f, spread %.1e" % (E[0], E.max() - E.min()))
print("L: initial %.6f, spread %.1e" % (L[0], L.max() - L.min()))
print("A: initial (%.6f, %.6f), spread %.1e" % (Ax[0], Ay[0], max(Ax.max() - Ax.min(), Ay.max() - Ay.min())))
e = np.hypot(Ax[0], Ay[0]); p = L[0]**2
phi = np.arctan2(y, x)
print("e = |A| = %.6f, p = L^2 = %.6f, e^2 = 1 + 2 E L^2: %s" % (e, p, abs(e**2 - (1 + 2 * E[0] * L[0]**2)) < 1e-9))
print("max |r (1 + e cos(phi - pi)) - p| along the orbit: %.1e" % np.max(np.abs(r * (1 + e * np.cos(phi - np.pi)) - p)))
print("semi-major axis p/(1 - e^2) = %.6f, 1/(2|E|) = %.6f" % (p / (1 - e**2), 1 / (2 * abs(E[0]))))

# Exercise exr-sr-cyclotron-numeric: integration of m du/dtau = q F u for a uniform magnetic field (m = q = B = 1) with
# initial gamma = 3; periods in coordinate and proper time.
import numpy as np
from scipy.integrate import solve_ivp
F = np.zeros((4, 4)); F[1, 2] = -1; F[2, 1] = 1          # F^{mu}_{nu} for B along z: du^x/dtau = B u^y, du^y/dtau = -B u^x
def rhs(tau, Y):
    u = Y[4:]; return np.concatenate([u, F @ u])
g = 3.0; v = np.sqrt(1 - 1/g**2)
Y0 = np.array([0, 0, 0, 0, g, g*v, 0, 0])
sol = solve_ivp(rhs, [0, 4*np.pi], Y0, rtol=1e-11, atol=1e-12, dense_output=True)
taus = np.linspace(0, 4*np.pi, 40001); Ys = sol.sol(taus)
phi = np.unwrap(np.arctan2(Ys[6], Ys[5])); k = np.argmax(np.abs(phi - phi[0]) >= 2*np.pi)
tau_period = np.interp(2*np.pi, np.abs(phi[k-1:k+1] - phi[0]), taus[k-1:k+1])
print("proper period %.6f (2 pi m/(qB) = %.6f)" % (tau_period, 2*np.pi))
print("coordinate period %.6f (2 pi gamma m/(qB) = %.6f)" % (sol.sol(tau_period)[0], 2*np.pi*g))
print("u.u stays 1: max deviation %.1e" % np.max(np.abs(sol.y[4]**2 - sol.y[5]**2 - sol.y[6]**2 - sol.y[7]**2 - 1)))

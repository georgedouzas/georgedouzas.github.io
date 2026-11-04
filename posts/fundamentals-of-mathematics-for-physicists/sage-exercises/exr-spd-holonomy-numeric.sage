# Exercise exr-spd-holonomy-numeric: numerical spinor transport around latitude circles of the unit sphere compared with
# exp(-i A sigma_3 / 2), A = 2 pi (1 - cos theta_0), and the curvature integral over the cap.
import numpy as np
from scipy.integrate import solve_ivp
for th0 in (0.3, 1.0, float(pi / 2), 2.5):
    f = lambda phi, y: [0.5j * np.cos(th0) * y[0], -0.5j * np.cos(th0) * y[1]]
    sol = solve_ivp(f, [0, 2 * np.pi], [1 + 0j, 1 + 0j], rtol=1e-11, atol=1e-12)
    H = (-sol.y[0, -1], -sol.y[1, -1]); A = 2 * np.pi * (1 - np.cos(th0))
    print("theta_0 = %.4f: H = diag(%s, %s); exp(-iA/2) = %s; error %.1e" % (th0, np.round(H[0], 8), np.round(H[1], 8), np.round(np.exp(-0.5j * A), 8), abs(H[0] - np.exp(-0.5j * A)) + abs(H[1] - np.exp(0.5j * A))))
t, t0 = var("t t0")
print("int over the cap of (1/2) R_12 = (1/2) int_0^t0 int_0^2pi sin(t) dphi dt =", (integrate(sin(t), t, 0, t0, algorithm="sympy") * 2 * pi / 2).simplify_full())

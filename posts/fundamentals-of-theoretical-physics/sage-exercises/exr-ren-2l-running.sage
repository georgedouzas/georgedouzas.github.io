# Exercise exr-ren-2l-running: one-loop and two-loop running of u = lambda/(16 pi^2) from u = 0.01 over t = log(mu/mu0) in [0, 25].
from scipy.integrate import solve_ivp
f1 = lambda t, u: [3 * u[0]**2]
f2 = lambda t, u: [3 * u[0]**2 - 17 / 3 * u[0]**3]
s1 = solve_ivp(f1, [0, 25], [0.01], rtol=1e-11, atol=1e-14); s2 = solve_ivp(f2, [0, 25], [0.01], rtol=1e-11, atol=1e-14)
u1, u2 = s1.y[0][-1], s2.y[0][-1]
print("one loop: u(25) = %.6f (exact 0.01/(1 - 0.75) = %.6f)" % (u1, 0.01 / 0.25))
print("two loops: u(25) = %.6f, relative difference %.3f%%" % (u2, 100 * (u1 - u2) / u1))
print("one-loop Landau pole at t = 1/(3 * 0.01) = %.2f" % (1 / 0.03))

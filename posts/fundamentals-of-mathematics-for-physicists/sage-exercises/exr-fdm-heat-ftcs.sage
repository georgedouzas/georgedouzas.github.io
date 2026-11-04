# Exercise exr-fdm-heat-ftcs (Foulaadvand, Secs. 5.3-5.4, eqs. (5.33)-(5.49), Figs. 5.4-5.5): a rod on [-1/2, 1/2] with
# u = 10 and 30 at the ends, diffusivity 1, initial data the discrete delta at the centre (height 1/dx), 60 grid
# intervals. FTCS with r = tau/dx^2 = 0.24 and 0.53, then Crank-Nicolson with r = 5; distance to the steady profile
# 20 + 20x and the largest value reached.
import numpy as np
M = 60; dx = 1.0/M; x = np.linspace(-0.5, 0.5, M + 1)
steady = 20 + 20*x
def initial():
    u = np.zeros(M + 1); u[M//2] = 1/dx; u[0], u[-1] = 10.0, 30.0
    return u
def ftcs(r, steps):
    u = initial(); big = 0
    for n in range(steps):
        u[1:-1] = u[1:-1] + r*(u[2:] - 2*u[1:-1] + u[:-2]); big = max(big, np.abs(u).max())
    return u, big
for r, times in [(0.24, [0.01, 0.05, 0.3, 1.0]), (0.53, [0.005, 0.01, 0.02])]:
    for t_end in times:
        steps = int(round(t_end/(r*dx^2)))
        u, big = ftcs(r, steps)
        print("FTCS r = %.2f, t = %.3f (%5d steps): max|u - steady| = %.3g" % (r, t_end, steps, np.abs(u - steady).max()))
# Crank-Nicolson: (1 + r) u_i^{n+1} - (r/2)(u_{i+1}^{n+1} + u_{i-1}^{n+1}) = (1 - r) u_i^n + (r/2)(u_{i+1}^n + u_{i-1}^n)
r = 5.0; n_in = M - 1
A = np.diag((1 + r)*np.ones(n_in)) + np.diag(-r/2*np.ones(n_in - 1), 1) + np.diag(-r/2*np.ones(n_in - 1), -1)
for t_end in [0.05, 0.3, 1.0]:
    steps = int(round(t_end/(r*dx^2))); u = initial()
    for n in range(steps):
        rhs = (1 - r)*u[1:-1] + r/2*(u[2:] + u[:-2]); rhs[0] += r/2*10.0; rhs[-1] += r/2*30.0   # the implicit boundary values
        u[1:-1] = np.linalg.solve(A, rhs)
    print("Crank-Nicolson r = %.0f, t = %.2f (%d steps): max|u - steady| = %.3g" % (r, t_end, steps, np.abs(u - steady).max()))

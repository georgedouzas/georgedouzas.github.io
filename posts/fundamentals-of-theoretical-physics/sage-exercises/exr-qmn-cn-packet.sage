# Exercise exr-qmn-cn-packet (Izaac-Wang, Secs. 11.2-11.2.2 and Problems P11.2-P11.3): a free Gaussian packet with
# x0 = 25, p0 = 2, sigma = 2 (units hbar = m = 1) on the grid [0, 100], dx = 0.05, propagated with the Crank-Nicolson
# step (1 + i H dt/2) psi_new = (1 - i H dt/2) psi_old and with the explicit Euler step psi_new = (1 - i H dt) psi_old,
# dt = 0.005, H = -(1/2) d^2/dx^2 by second-order finite differences. The norm, the mean and the width are compared with
# the exact values <x> = x0 + p0 t and sigma(t) = sigma sqrt(1 + (t/2 sigma^2)^2) of prp-qm-box-free.
import numpy as np
from scipy.linalg import solve_banded
dx, dt = 0.05, 0.005
x = np.arange(dx, 100, dx); n = len(x)
psi0 = (2*np.pi*4)^(-0.25)*np.exp(-(x - 25)^2/16 + 2j*x)
off = -0.5/dx^2*np.ones(n); dia = 1/dx^2*np.ones(n)
def Hmul(p):
    q = dia*p; q[1:] += off[1:]*p[:-1]; q[:-1] += off[:-1]*p[1:]; return q
ab = np.array([0.5j*dt*off, 1 + 0.5j*dt*dia, 0.5j*dt*off])
def moments(p):
    w = np.abs(p)^2*dx; N = w.sum(); m = (w*x).sum()/N
    return N, m, np.sqrt((w*(x - m)^2).sum()/N)
pc, pe = psi0.copy(), psi0.copy()
for step in range(1, 4001):
    pc = solve_banded((1, 1), ab, pc - 0.5j*dt*Hmul(pc))
    if step <= 60:
        pe = pe - 1j*dt*Hmul(pe)
        if step in (5, 20, 60):
            print("t = %5.3f Euler: norm %.3e" % (step*dt, moments(pe)[0]))
    if step in (1400, 2000, 4000):
        t = step*dt; N, m, s = moments(pc)
        print("t = %4.1f  Crank-Nicolson: norm %.10f, <x> = %.4f (exact %.4f), width %.4f (exact %.4f)" % (t, N, m, 25 + 2*t, s, 2*sqrt(1 + (t/8)^2)))
print("on the grid the group velocity of p0 = 2 is sin(p0 dx)/dx = %.4f, so <x> lags by %.3f at t = 20" % (sin(2*dx)/dx, 20*(2 - sin(2*dx)/dx)))
print("Euler growth factor per step for the highest grid mode, sqrt(1 + (E dt)^2) with E = 2/dx^2: %.6f" % sqrt(1 + (2/dx^2*dt)^2))

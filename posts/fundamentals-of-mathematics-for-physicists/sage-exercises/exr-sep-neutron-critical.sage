# Exercise exr-sep-neutron-critical (Foulaadvand, Sec. 5.5, eqs. (5.53)-(5.69), Fig. 5.8, and Problem 5.7): the
# density n_t = D n_xx + C n in units C = D = 1 on [-L/2, L/2], absorbed at the ends (n = 0), from a central delta.
# FTCS (r = 0.25, 60 intervals) gives the late-time growth rate of the total sum(n) dx, compared with the
# separation constant alpha_1 = 1 - (pi/L)^2; the critical length is pi. With n = 0 at the left end and n_x = 0
# at the right end the lowest mode is sin(pi(x + L/2)/(2L)), alpha_1 = 1 - (pi/(2L))^2, critical length pi/2.
import numpy as np
def rate(L, mixed=False, M=60, r=0.25, t1=3.0, t2=4.0):
    dx = L/M; tau = r*dx^2
    n = np.zeros(M + 1); n[M//2] = 1/dx
    totals = {}
    steps2 = int(round(t2/tau))
    for k in range(1, steps2 + 1):
        end = n[-1] + 2*r*(n[-2] - n[-1]) + tau*n[-1]   # n_x = 0 by a mirror point n_{M+1} = n_{M-1}
        n[1:-1] = n[1:-1] + r*(n[2:] - 2*n[1:-1] + n[:-2]) + tau*n[1:-1]
        n[0] = 0.0
        n[-1] = end if mixed else 0.0
        if k in (int(round(t1/tau)), steps2):
            totals[k] = n.sum()*dx
    (k1, s1), (k2, s2) = sorted(totals.items())
    return np.log(s2/s1)/((k2 - k1)*tau)
for L in [2.5, 3.0, 3.3, 3.5]:
    print("Dirichlet, L = %.1f: growth rate %+.4f, alpha_1 = %+.4f" % (L, rate(L), 1 - (np.pi/L)^2))
for L in [1.4, 1.7]:
    print("mixed,     L = %.1f: growth rate %+.4f, alpha_1 = %+.4f" % (L, rate(L, mixed=True), 1 - (np.pi/(2*L))^2))
print("critical lengths: pi = %.4f (both ends absorbing), pi/2 = %.4f (one reflecting end)" % (np.pi, np.pi/2))

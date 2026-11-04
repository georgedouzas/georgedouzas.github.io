# Exercise exr-vcp-add-kkcount: the KK gravitons of a torus of radius R_t have masses |k|/R_t, k in Z^n. The number below
# E is the number of lattice points in a ball of radius E R_t, close to the volume Omega_n (E R_t)^n with
# Omega_n = pi^{n/2}/Gamma(n/2 + 1). Each couples with 1/M_Pl, so the summed rate is ~ (E R_t)^n/M_Pl^2.
from itertools import product
for n in [1, 2, 3]:
    for ER in [5, 10, 20]:
        cnt = sum(1 for v in product(range(-ER, ER + 1), repeat=n) if sum(x * x for x in v) <= ER * ER)
        vol = (pi**(n / 2) / gamma(n / 2 + 1) * ER**n).n()
        print("n = %d, E R_t = %d: %d modes, Omega_n (E R_t)^n = %.1f, ratio %.3f" % (n, ER, cnt, vol, cnt / vol))

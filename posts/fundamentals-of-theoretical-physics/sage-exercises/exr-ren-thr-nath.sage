# Exercise exr-ren-thr-nath: M_H3 from 3/alpha_2 - 2/alpha_3 - 1/alpha_1 = (1/2pi)((12/5) log(M_H3/M_Z) - 2 log(M_S/M_Z)) (Nath eq. 9.92).
MZ = 91.1876; a1, a2 = 0.016985, 0.03358
for MS in [1e3, 1e4]:
    for a3 in [0.111, 0.118, 0.125]:
        lhs = 3 / a2 - 2 / a3 - 1 / a1
        logM = 5 / 12 * (2 * pi * lhs + 2 * log(MS / MZ))
        print("M_S = %.0e, alpha_3 = %.3f: left side %.3f, M_H3 = %.2e GeV" % (MS, a3, lhs, (MZ * exp(logM)).n()))

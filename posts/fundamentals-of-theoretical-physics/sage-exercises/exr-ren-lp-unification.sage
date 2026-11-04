# Exercise exr-ren-lp-unification: pairwise crossings of the one-loop inverse couplings, SM and MSSM.
MZ = 91.1876; ainv = [1 / 0.016985, 1 / 0.03358, 1 / 0.118]
for name, b in [("SM", [41 / 10, -19 / 6, -7]), ("MSSM", [33 / 5, 1, -3])]:
    for (i, j) in [(0, 1), (1, 2), (0, 2)]:
        t = (ainv[i] - ainv[j]) / ((b[i] - b[j]) / (2 * pi))
        print("%-4s alpha_%d = alpha_%d at mu = %.2e GeV, alpha^-1 = %.2f" % (name, i + 1, j + 1, (MZ * exp(t)).n(), (ainv[i] - b[i] / (2 * pi) * t).n()))

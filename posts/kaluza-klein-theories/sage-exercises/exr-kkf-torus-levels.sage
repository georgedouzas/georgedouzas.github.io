# Exercise exr-kkf-torus-levels: the five lightest four-dimensional mass levels of a six-dimensional Dirac fermion on
# T^2 with R_1 = 1, R_2 = 1/2 for the four spin structures, with the number of four-dimensional Dirac fermions at each
# level (N = 2 per lattice point, N = 2^floor(n/2)).
half = QQ(1) / 2
R = (1, half)
for al, name in [((0, 0), "(P, P)"), ((half, 0), "(A, P)"), ((0, half), "(P, A)"), ((half, half), "(A, A)")]:
    lev = {}
    for m1 in range(-10, 11):
        for m2 in range(-10, 11):
            s = ((m1 + al[0]) / R[0]) ** 2 + ((m2 + al[1]) / R[1]) ** 2
            lev[s] = lev.get(s, 0) + 1
    low = sorted(lev)[:5]
    print(name, "  mass^2:", low, "  Dirac fermions:", [2 * lev[s] for s in low])

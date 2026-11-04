# Exercise exr-ls-torus-rectangular: the ten lowest eigenvalues m^2 + n^2/rho of the torus with R_1 = 1, R_2^2 = rho and
# their multiplicities, for rho = 1, 2, pi; accidental degeneracies for rational rho, none for irrational rho.
from collections import Counter
for rho in [1, 2, pi]:
    ev = Counter()
    for m in range(-8, 9):
        for n in range(-8, 9):
            ev[m ** 2 + n ** 2 / rho] += 1
    lowest = sorted(ev.items(), key=lambda t: t[0].n() if hasattr(t[0], "n") else t[0])[:10]
    print("rho =", rho, ":", [(lam, mult) for lam, mult in lowest])
print("rho = 2: (2, 1) and (0, 3) both give", 4 + 1 / QQ(2), "=", 0 + 9 / QQ(2), "-> multiplicity 4 + 2 = 6; likewise (3, 0) and (1, 4) give", 9, "=", 1 + 16 / QQ(2))

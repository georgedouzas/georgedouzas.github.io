# Exercise exr-spd-torus-lowest: the six smallest |k| on the square torus (R = 1) for the four spin structures, with the
# number of lattice points; each lattice point gives the two eigenvalues +-|k| (N = 2).
from collections import Counter
for name, al in (("(P,P)", (0, 0)), ("(A,P)", (1/2, 0)), ("(A,P)'", (0, 1/2)), ("(A,A)", (1/2, 1/2))):
    c = Counter()
    for m1 in range(-6, 7):
        for m2 in range(-6, 7):
            c[(m1 + al[0]) ^ 2 + (m2 + al[1]) ^ 2] += 1
    low = sorted(c)[:6]
    print(name.ljust(7), ", ".join("|k| = %s: %d" % (sqrt(QQ(q)), c[q]) for q in low), "| zero modes:", 2 * c[0])

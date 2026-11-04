# Exercise exr-spd-instanton-check: self-duality of F^i = e^i ^ e^4 + (1/2) eps_ijk e^j ^ e^k on S^4 in an orthonormal
# frame, and the instanton index (2/3) J (J+1)(2J+1) against Dolan's zero-mode count (2I+1)(2I+2)(2I+3)/6, I = J - 1/2.
def sgn(*p): return Permutation([int(a) + 1 for a in p]).sign() if len(set(p)) == len(p) else 0
def two_form(i):
    # components F_ab of F = (1/2) F_ab e^a ^ e^b (indices 0..3, the fourth direction is index 3)
    F = matrix(QQ, 4, 4)
    F[i, 3], F[3, i] = 1, -1
    for j in range(3):
        for k in range(3):
            F[j, k] += sgn(i, j, k)
    return F
star = lambda F: matrix(QQ, 4, 4, lambda a, b: sum(sgn(a, b, c, d) * F[c, d] for c in range(4) for d in range(4)) / 2)
for i in range(3):
    F = two_form(i)
    print("F^%d components:" % (i + 1), F.rows(), " self-dual:", star(F) == F)
for twoJ in range(1, 7):
    J = QQ(twoJ) / 2; Ii = J - 1/2
    print("J = %s: index %s, zero modes (2I+1)(2I+2)(2I+3)/6 = %s" % (J, 2/3 * J * (J + 1) * (2 * J + 1), (2 * Ii + 1) * (2 * Ii + 2) * (2 * Ii + 3) / 6))

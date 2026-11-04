# Exercise exr-spd-s3-matrices: D = 4 S.J + 3/2 on C^2 (x) V_j for 2j = 0..8, multiplicities times 2j + 1, compared with
# the sphere formula +-(l + 3/2), multiplicity (l + 1)(l + 2).
from collections import Counter
def spin_matrices(j):
    j = QQ(j); dim = int(2 * j + 1); ms = [j - k for k in range(dim)]
    Jz = diagonal_matrix(QQbar, ms); Jp = matrix(QQbar, dim, dim)
    for k in range(1, dim):
        Jp[k - 1, k] = sqrt(QQbar(j * (j + 1) - ms[k] * (ms[k] + 1)))
    Jm = Jp.transpose()
    return (Jp + Jm) / 2, (Jp - Jm) / (2 * QQbar(I)), Jz
S = spin_matrices(1/2)
total = Counter()
for twoj in range(0, 9):
    J = spin_matrices(twoj / 2)
    D = 4 * sum(S[a].tensor_product(J[a]) for a in range(3)) + 3/2 * identity_matrix(QQbar, 2 * (twoj + 1))
    ev = Counter(QQ(e) for e in D.eigenvalues())
    for e, m in ev.items(): total[e] += m * (twoj + 1)
    print("j = %s: %s" % (twoj / 2, dict(sorted(ev.items()))))
ok = all(total[l + 3/2] == (l + 1) * (l + 2) and total[-(l + 3/2)] == (l + 1) * (l + 2) for l in range(0, 8))
print("multiplicities of +-(l + 3/2) for l <= 7 equal (l + 1)(l + 2):", ok)

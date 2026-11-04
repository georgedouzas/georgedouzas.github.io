# Exercise exr-spd-spin-orbit-matrices: sigma.L + 1 = 2 S.L + 1 on C^2 (x) C^{2l+1}; eigenvalues l + 1 (2l + 2 times) and
# -l (2l times), giving the Dirac spectrum +-(k + 1) of S^2 with multiplicity 2k + 2.
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
for l in range(5):
    L = spin_matrices(l)
    K = sum(2 * S[a].tensor_product(L[a]) for a in range(3)) + identity_matrix(QQbar, 2 * (2 * l + 1))
    ev = Counter(ZZ(e) for e in K.eigenvalues())
    total.update(ev)
    print("l = %d:" % l, dict(sorted(ev.items())))
print("collected over l <= 4 (eigenvalue: multiplicity):", dict(sorted((k, m) for k, m in total.items() if abs(k) <= 4)))

# Exercise exr-vcr-fz-casimir: functions on the fuzzy sphere. For the N-dimensional irreducible representation J_a of
# su(2), the N x N matrices decompose under the adjoint action X -> [J_a, X] into spins l = 0, ..., N-1: the operator
# C(X) = sum_a [J_a, [J_a, X]] on Mat(N) has eigenvalues l(l+1) with multiplicities 2l+1, a truncated set of spherical
# harmonics. Check for N = 2, 3, 4, together with the Casimir J_a J_a = (N^2 - 1)/4.
def spin(N):
    j = QQ(N - 1) / 2; ms = [j - k for k in range(N)]
    Jp = matrix(QQbar, N, N, lambda a, b: sqrt(j * (j + 1) - ms[b] * (ms[b] + 1)) if a == b - 1 else 0)
    Jm = Jp.transpose()
    return [(Jp + Jm) / 2, (Jp - Jm) / (2 * QQbar(I)), diagonal_matrix(QQbar, ms)]
for N in [2, 3, 4]:
    J = spin(N)
    cas = sum(Ji * Ji for Ji in J)
    basis = []
    for i in range(N):
        for k in range(N):
            E = matrix(QQbar, N, N); E[i, k] = 1; basis.append(E)
    C = matrix(QQbar, [sum((Ja * (Ja * E - E * Ja) - (Ja * E - E * Ja) * Ja) for Ja in J).list() for E in basis]).transpose()
    ev = C.eigenvalues()
    print("N = %d: J.J = %s x 1; adjoint Casimir eigenvalues with multiplicities %s" % (N, QQ(cas[0, 0]), sorted((QQ(e), ev.count(e)) for e in set(ev))))

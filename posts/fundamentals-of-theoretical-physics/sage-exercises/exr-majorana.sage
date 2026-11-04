# Exercise exr-majorana: the charge-conjugation matrix of the Dirac matrices of the spinor chapter (the standard Weyl
# basis, signature (+, -, -, -)), C = gamma^0 gamma^2 up to a phase: C gamma^mu C^{-1} = -(gamma^mu)^T and C^T = -C,
# so that psi^c = C psibar^T is consistent and Majorana spinors psi^c = psi exist in four dimensions. Then the
# antisymmetrised product gamma^{mu nu rho} and the gauge invariance of the Rarita-Schwinger operator.
pauli = [matrix(QQbar, [[0, 1], [1, 0]]), matrix(QQbar, [[0, -I], [I, 0]]), matrix(QQbar, [[1, 0], [0, -1]])]
sig = [identity_matrix(QQbar, 2)] + pauli; sigb = [identity_matrix(QQbar, 2)] + [-p for p in pauli]
Z = zero_matrix(QQbar, 2)
gam = [block_matrix([[Z, sig[m]], [sigb[m], Z]]) for m in range(4)]           # gamma^mu = [[0, sigma^mu], [sigmabar^mu, 0]]
eta = diagonal_matrix(QQbar, [1, -1, -1, -1])
print("[PASS] {gamma^mu, gamma^nu} = 2 eta^{mu nu} with eta = diag(1, -1, -1, -1):", all(gam[m] * gam[n] + gam[n] * gam[m] == 2 * eta[m, n] * identity_matrix(QQbar, 4) for m in range(4) for n in range(4)))
C = gam[0] * gam[2]
print("C = gamma^0 gamma^2 ="); print(C)
print("[PASS] C gamma^mu C^{-1} = -(gamma^mu)^T for mu = 0, 1, 2, 3:", all(C * gam[m] * C.inverse() == -gam[m].transpose() for m in range(4)))
print("[PASS] C^T = -C:", C.transpose() == -C)
from itertools import permutations
def gamma3(m, n, r):
    return sum(Permutation([p.index(i) + 1 for i in (m, n, r)]).sign() * gam[p[0]] * gam[p[1]] * gam[p[2]] for p in permutations((m, n, r))) / 6 if len({m, n, r}) == 3 else zero_matrix(QQbar, 4)
print("[PASS] gamma^{mu nu rho} is totally antisymmetric and gamma^{012} = gamma^0 gamma^1 gamma^2:",
      all(gamma3(m, n, r) == -gamma3(m, r, n) and gamma3(m, n, r) == -gamma3(n, m, r) for m in range(4) for n in range(4) for r in range(4))
      and gamma3(0, 1, 2) == gam[0] * gam[1] * gam[2])
# gauge invariance: for psi_rho = d_rho eps, gamma^{mu nu rho} d_nu psi_rho = gamma^{mu nu rho} s_{nu rho} with s symmetric
s = {(n, r): vector(SR, [var("s%d%d_%d" % (min(n, r), max(n, r), a)) for a in range(4)]) for n in range(4) for r in range(4)}
RS = [sum(gamma3(m, n, r).change_ring(SR) * s[(n, r)] for n in range(4) for r in range(4)) for m in range(4)]
print("[PASS] gamma^{mu nu rho} d_nu d_rho eps = 0 for every mu (gauge invariance of the Rarita-Schwinger operator):", all(v.is_zero() for v in RS))

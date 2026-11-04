# Exercise exr-ssg-lorentz-generators: the Lorentz generators on left-handed Weyl spinors,
# S^{mu nu} = (i/4)(sigma^mu sigmabar^nu - sigma^nu sigmabar^mu), satisfy the Lorentz algebra in the metric (+, -, -, -),
# [S^{mu nu}, S^{rho sigma}] = i(eta^{nu rho} S^{mu sigma} - eta^{mu rho} S^{nu sigma} - eta^{nu sigma} S^{mu rho} + eta^{mu sigma} S^{nu rho}),
# the rotations S^{ij} = eps_{ijk} sigma_k/2 are Hermitian and the boosts S^{0i} = -i sigma_i/2 anti-Hermitian; the
# right-handed generators (i/4)(sigmabar sigma - ...) give the same algebra.
pauli = [matrix(QQbar, [[0, 1], [1, 0]]), matrix(QQbar, [[0, -I], [I, 0]]), matrix(QQbar, [[1, 0], [0, -1]])]
sig = [identity_matrix(QQbar, 2)] + pauli; sigb = [identity_matrix(QQbar, 2)] + [-p for p in pauli]
eta = [1, -1, -1, -1]
def e(m, n): return eta[m] if m == n else 0
SL = [[I / 4 * (sig[m] * sigb[n] - sig[n] * sigb[m]) for n in range(4)] for m in range(4)]
SR_ = [[I / 4 * (sigb[m] * sig[n] - sigb[n] * sig[m]) for n in range(4)] for m in range(4)]
def algebra(S):
    return all(S[m][n] * S[r][s] - S[r][s] * S[m][n] == I * (e(n, r) * S[m][s] - e(m, r) * S[n][s] - e(n, s) * S[m][r] + e(m, s) * S[n][r])
               for m in range(4) for n in range(4) for r in range(4) for s in range(4))
print("[PASS] the left-handed generators satisfy the Lorentz algebra with eta = diag(1, -1, -1, -1):", algebra(SL))
print("[PASS] so do the right-handed generators built from sigmabar sigma:", algebra(SR_))
print("[PASS] rotations S^{12} = sigma_3/2 (Hermitian), boosts S^{03} = -i sigma_3/2 (anti-Hermitian):",
      SL[1][2] == pauli[2] / 2 and SL[0][3] == -I * pauli[2] / 2)
print("[PASS] the boosts have opposite signs on the two chiralities, the rotations agree:",
      all(SR_[0][k] == -SL[0][k] for k in (1, 2, 3)) and all(SR_[i][j] == SL[i][j] for i in (1, 2, 3) for j in (1, 2, 3)))

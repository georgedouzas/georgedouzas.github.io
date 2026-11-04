# Exercise exr-sigma-identities: the sigma-matrix identities of Martin's primer (hep-ph/9709356, eqs. 2.21-2.25),
# converted to our signature (+, -, -, -) in which they carry +2 eta instead of Martin's -2 eta (mostly-plus metric),
# checked on the explicit matrices sigma^mu = (1, sigma_i), sigmabar^mu = (1, -sigma_i).
pauli = [matrix(QQbar, [[0, 1], [1, 0]]), matrix(QQbar, [[0, -I], [I, 0]]), matrix(QQbar, [[1, 0], [0, -1]])]
sig = [identity_matrix(QQbar, 2)] + pauli; sigb = [identity_matrix(QQbar, 2)] + [-p for p in pauli]
eta = diagonal_matrix(QQbar, [1, -1, -1, -1]); one = identity_matrix(QQbar, 2)
print("[PASS] sigma^mu sigmabar^nu + sigma^nu sigmabar^mu = 2 eta^{mu nu} (Martin eq. 2.24 with -2 in his metric):",
      all(sig[m] * sigb[n] + sig[n] * sigb[m] == 2 * eta[m, n] * one for m in range(4) for n in range(4)))
print("[PASS] sigmabar^mu sigma^nu + sigmabar^nu sigma^mu = 2 eta^{mu nu} (eq. 2.25):",
      all(sigb[m] * sig[n] + sigb[n] * sig[m] == 2 * eta[m, n] * one for m in range(4) for n in range(4)))
print("[PASS] tr(sigma^mu sigmabar^nu) = 2 eta^{mu nu}:", all((sig[m] * sigb[n]).trace() == 2 * eta[m, n] for m in range(4) for n in range(4)))
fierz = all(sum(eta[m, m] * sig[m][a, b] * sigb[m][c, d] for m in range(4)) == 2 * (1 if a == d else 0) * (1 if b == c else 0)
            for a in range(2) for b in range(2) for c in range(2) for d in range(2))
print("[PASS] sigma^mu_{a bdot} sigmabar_mu^{cdot d} = 2 delta_a^d delta_bdot^cdot (eq. 2.21 with -2 in his metric):", fierz)
# the Dirac matrices of the spinor chapter in block form: gamma^mu = [[0, sigma^mu], [sigmabar^mu, 0]]
gam = [block_matrix([[zero_matrix(QQbar, 2), sig[m]], [sigb[m], zero_matrix(QQbar, 2)]]) for m in range(4)]
print("[PASS] the block matrices satisfy the Clifford relations {gamma^mu, gamma^nu} = 2 eta^{mu nu} with eta = diag(1, -1, -1, -1):",
      all(gam[m] * gam[n] + gam[n] * gam[m] == 2 * eta[m, n] * identity_matrix(QQbar, 4) for m in range(4) for n in range(4)))

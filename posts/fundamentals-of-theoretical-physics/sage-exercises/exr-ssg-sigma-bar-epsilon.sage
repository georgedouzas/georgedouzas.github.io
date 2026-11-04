# Exercise exr-ssg-sigma-bar-epsilon: the matrices sigma^mu are Hermitian, and sigmabar^mu is obtained from sigma^mu by
# raising both indices with epsilon: (sigmabar^mu)^{adot a} = eps^{adot bdot} eps^{a b} sigma^mu_{b bdot}, with
# eps^{12} = -eps^{21} = 1 (Martin, hep-ph/9709356, eq. 2.20).
pauli = [matrix(QQbar, [[0, 1], [1, 0]]), matrix(QQbar, [[0, -I], [I, 0]]), matrix(QQbar, [[1, 0], [0, -1]])]
sig = [identity_matrix(QQbar, 2)] + pauli; sigb = [identity_matrix(QQbar, 2)] + [-p for p in pauli]
eps_up = matrix(QQbar, [[0, 1], [-1, 0]])            # eps^{12} = 1
print("[PASS] sigma^mu is Hermitian for every mu:", all(s.conjugate_transpose() == s for s in sig))
raised = [matrix(QQbar, 2, 2, lambda ad, a: sum(eps_up[ad, bd] * eps_up[a, b] * sig[m][b, bd] for b in range(2) for bd in range(2))) for m in range(4)]
print("[PASS] eps eps sigma^mu = sigmabar^mu, i.e. eps sigma^T eps^T = sigmabar:", all(raised[m] == sigb[m] for m in range(4)))
print("[PASS] for the spatial matrices this is sigma_2 sigma_i^T sigma_2 = -sigma_i:", all(pauli[1] * p.transpose() * pauli[1] == -p for p in pauli))

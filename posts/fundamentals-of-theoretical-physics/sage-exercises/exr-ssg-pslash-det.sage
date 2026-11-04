# Exercise exr-ssg-pslash-det: for a four-vector p_mu (lower index, metric (+, -, -, -)) the matrices p.sigma =
# p_mu sigma^mu and p.sigmabar = p_mu sigmabar^mu satisfy det(p.sigma) = p^2 = p_mu p^mu and
# (p.sigma)(p.sigmabar) = p^2 * 1; every Hermitian 2x2 matrix is p.sigma for a unique real p, the components being
# here recovered with the trace identity tr(sigma^mu sigmabar^nu) = 2 eta^{mu nu}.
p0, p1, p2, p3 = var("p0 p1 p2 p3", domain="real")
pauli = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
sig = [identity_matrix(SR, 2)] + pauli; sigb = [identity_matrix(SR, 2)] + [-q for q in pauli]
eta = diagonal_matrix([1, -1, -1, -1]); p = [p0, p1, p2, p3]              # lower-index components p_mu
psq = sum(eta[m, m] * p[m]^2 for m in range(4))                          # p_mu p^mu
ps = sum(p[m] * sig[m] for m in range(4)); psb = sum(p[m] * sigb[m] for m in range(4))
print("p.sigma ="); print(ps)
print("[PASS] det(p.sigma) = det(p.sigmabar) = p_mu p^mu:", bool((ps.det() - psq).expand() == 0) and bool((psb.det() - psq).expand() == 0))
print("[PASS] (p.sigma)(p.sigmabar) = (p.sigmabar)(p.sigma) = p^2 times the identity:",
      ((ps * psb).expand() - psq * identity_matrix(2)).is_zero() and ((psb * ps).expand() - psq * identity_matrix(2)).is_zero())
# recover p_mu from X = p.sigma: tr(X sigmabar^nu) = 2 p_mu eta^{mu nu} = 2 p^nu
print("[PASS] tr((p.sigma) sigmabar^nu) = 2 p^nu = 2 eta^{nu nu} p_nu for every nu:",
      all(bool(((ps * sigb[n]).trace() - 2 * eta[n, n] * p[n]).expand() == 0) for n in range(4)))
# a light-like momentum: p.sigma has rank one
E = var("E", domain="positive")
pl = [E, 0, 0, -E]                                                       # p^mu = (E, 0, 0, E), p_3 = -E
X = sum(pl[m] * sig[m] for m in range(4))
print("for p^mu = (E, 0, 0, E): p.sigma =", X.list(), " rank", X.rank())
print("[PASS] a null momentum gives a rank-one p.sigma = diag(0, 2E):", X == diagonal_matrix([0, 2 * E]) and X.rank() == 1)

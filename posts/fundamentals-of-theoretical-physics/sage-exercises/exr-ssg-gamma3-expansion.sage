# Exercise exr-ssg-gamma3-expansion: the antisymmetrised product gamma^{mu nu rho} (weight one) in four dimensions with
# eta = diag(1, -1, -1, -1) satisfies gamma^mu gamma^nu gamma^rho = gamma^{mu nu rho} + eta^{mu nu} gamma^rho - eta^{mu rho} gamma^nu
# + eta^{nu rho} gamma^mu, and in four dimensions gamma^{mu nu rho} = c eps^{mu nu rho sigma} gamma_sigma gamma^5 with a constant c
# determined below (eps_{0123} = +1, so eps^{0123} = -1; gamma^5 = i gamma^0 gamma^1 gamma^2 gamma^3).
from itertools import permutations
pauli = [matrix(QQbar, [[0, 1], [1, 0]]), matrix(QQbar, [[0, -I], [I, 0]]), matrix(QQbar, [[1, 0], [0, -1]])]
sig = [identity_matrix(QQbar, 2)] + pauli; sigb = [identity_matrix(QQbar, 2)] + [-p for p in pauli]
Z = zero_matrix(QQbar, 2); gam = [block_matrix([[Z, sig[m]], [sigb[m], Z]]) for m in range(4)]
eta = [1, -1, -1, -1]; e = lambda a, b: eta[a] if a == b else 0
def g3(m, n, r):
    if len({m, n, r}) < 3: return zero_matrix(QQbar, 4)
    return sum(Permutation([p.index(i) + 1 for i in (m, n, r)]).sign() * gam[p[0]] * gam[p[1]] * gam[p[2]] for p in permutations((m, n, r))) / 6
print("[PASS] gamma^mu gamma^nu gamma^rho = gamma^{mu nu rho} + eta^{mu nu} gamma^rho - eta^{mu rho} gamma^nu + eta^{nu rho} gamma^mu for all indices:",
      all(gam[m] * gam[n] * gam[r] == g3(m, n, r) + e(m, n) * gam[r] - e(m, r) * gam[n] + e(n, r) * gam[m] for m in range(4) for n in range(4) for r in range(4)))
g5 = I * gam[0] * gam[1] * gam[2] * gam[3]
def eps_up(*idx): return -Permutation([i + 1 for i in idx]).sign() if len(set(idx)) == 4 else 0
dual = lambda m, n, r: sum(eps_up(m, n, r, s) * eta[s] * gam[s] * g5 for s in range(4))   # eps^{mu nu rho sigma} gamma_sigma gamma^5
cs = [c for c in (1, -1, I, -I) if all(g3(m, n, r) == c * dual(m, n, r) for m in range(4) for n in range(4) for r in range(4))]
print("gamma^{mu nu rho} = c eps^{mu nu rho sigma} gamma_sigma gamma^5 with c =", cs)
print("[PASS] the three-index gamma is dual to gamma_sigma gamma^5 with c = -i, so the Rarita-Schwinger operator is eps^{mu nu rho sigma} gamma^5 gamma_sigma d_nu psi_rho up to a factor:", cs == [-I])

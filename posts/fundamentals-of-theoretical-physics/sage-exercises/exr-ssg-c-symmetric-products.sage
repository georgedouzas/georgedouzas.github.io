# Exercise exr-ssg-c-symmetric-products: with C = gamma^0 gamma^2 (standard Weyl basis, signature (+, -, -, -)) the
# matrices C Gamma are symmetric or antisymmetric according to Gamma. We check that
# C, C gamma^5, C gamma^mu gamma^5 are antisymmetric and C gamma^mu, C gamma^{mu nu} symmetric. For anticommuting Majorana
# spinors (psibar = psi^T K with K = (C^T)^{-1}), the symmetry of K Gamma decides the flip rule
# psibar_1 Gamma psi_2 = +- psibar_2 Gamma psi_1: + for 1, gamma^5, gamma^mu gamma^5 and - for gamma^mu, gamma^{mu nu}.
pauli = [matrix(QQbar, [[0, 1], [1, 0]]), matrix(QQbar, [[0, -I], [I, 0]]), matrix(QQbar, [[1, 0], [0, -1]])]
sig = [identity_matrix(QQbar, 2)] + pauli; sigb = [identity_matrix(QQbar, 2)] + [-p for p in pauli]
Z = zero_matrix(QQbar, 2)
gam = [block_matrix([[Z, sig[m]], [sigb[m], Z]]) for m in range(4)]
g5 = I * gam[0] * gam[1] * gam[2] * gam[3]
C = gam[0] * gam[2]
print("gamma^5 =", g5.diagonal())
def sym(M): return "symmetric" if M.transpose() == M else "antisymmetric" if M.transpose() == -M else "neither"
sets = {"1": [identity_matrix(QQbar, 4)], "gamma^5": [g5], "gamma^mu gamma^5": [gam[m] * g5 for m in range(4)],
        "gamma^mu": gam, "gamma^{mu nu}": [(gam[m] * gam[n] - gam[n] * gam[m]) / 2 for m in range(4) for n in range(m + 1, 4)]}
res = {k: set(sym(C * G) for G in v) for k, v in sets.items()}
print(res)
print("[PASS] C, C gamma^5, C gamma^mu gamma^5 are antisymmetric; C gamma^mu and C gamma^{mu nu} are symmetric:",
      res == {"1": {"antisymmetric"}, "gamma^5": {"antisymmetric"}, "gamma^mu gamma^5": {"antisymmetric"}, "gamma^mu": {"symmetric"}, "gamma^{mu nu}": {"symmetric"}})
print("[PASS] counting: 6 antisymmetric (1 + 1 + 4) and 10 symmetric (4 + 6) of the 16 = 6 + 10 matrices C Gamma:",
      sum(len(v) for k, v in sets.items() if res[k] == {"antisymmetric"}) == 6 and sum(len(v) for k, v in sets.items() if res[k] == {"symmetric"}) == 10)
# flip rule: for Majorana psi = C psibar^T, psibar = psi^T C^{-1}... with C^{-1} = -C^T... the bilinear psibar_1 Gamma psi_2 = psi_1^T K Gamma psi_2
# with K = (C^T)^{-1} up to sign; anticommuting components give psi_1^T M psi_2 = -psi_2^T M^T psi_1
K = (C.transpose()).inverse()
flip = {k: set("+" if (K * G).transpose() == -(K * G) else "-" if (K * G).transpose() == (K * G) else "?" for G in v) for k, v in sets.items()}
print("flip signs psibar_1 Gamma psi_2 = (sign) psibar_2 Gamma psi_1:", flip)
print("[PASS] Majorana flip: + for 1, gamma^5, gamma^mu gamma^5 and - for gamma^mu, gamma^{mu nu} (anticommuting components):",
      flip == {"1": {"+"}, "gamma^5": {"+"}, "gamma^mu gamma^5": {"+"}, "gamma^mu": {"-"}, "gamma^{mu nu}": {"-"}})

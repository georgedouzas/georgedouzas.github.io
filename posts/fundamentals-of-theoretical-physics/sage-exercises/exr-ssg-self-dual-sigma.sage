# Exercise exr-ssg-self-dual-sigma: the gaugino variation contains sigma^mu sigmabar^nu F_{mu nu} = 4 sigma^{mu nu} F_{mu nu} with
# sigma^{mu nu} = (sigma^mu sigmabar^nu - sigma^nu sigmabar^mu)/4. These six matrices span only three dimensions: they are
# eigenvectors of the Hodge dual, (1/2) eps^{mu nu rho sigma} sigma_{rho sigma} = c sigma^{mu nu} with c = -+ i, eps_{0123} = +1, in the metric
# (+, -, -, -) up to the sign fixed below, so delta lambda sees only one duality component of F_{mu nu}; the sigmabar^{mu nu} have the other.
pauli = [matrix(QQbar, [[0, 1], [1, 0]]), matrix(QQbar, [[0, -I], [I, 0]]), matrix(QQbar, [[1, 0], [0, -1]])]
sig = [identity_matrix(QQbar, 2)] + pauli; sigb = [identity_matrix(QQbar, 2)] + [-q for q in pauli]
eta = [1, -1, -1, -1]
S = [[(sig[m] * sigb[n] - sig[n] * sigb[m]) / 4 for n in range(4)] for m in range(4)]
Sb = [[(sigb[m] * sig[n] - sigb[n] * sig[m]) / 4 for n in range(4)] for m in range(4)]
low = lambda T: [[eta[m] * eta[n] * T[m][n] for n in range(4)] for m in range(4)]       # lower both indices
def eps_up(*idx):                                                                    # eps^{0123} = -1
    return -Permutation([i + 1 for i in idx]).sign() if len(set(idx)) == 4 else 0
def dual(T):
    Tl = low(T)
    return [[sum(eps_up(m, n, r, s) * Tl[r][s] for r in range(4) for s in range(4)) / 2 for n in range(4)] for m in range(4)]
dS = dual(S); dSb = dual(Sb)
c_S = [c for c in (I, -I) if all(dS[m][n] == c * S[m][n] for m in range(4) for n in range(4))]
c_Sb = [c for c in (I, -I) if all(dSb[m][n] == c * Sb[m][n] for m in range(4) for n in range(4))]
print("dual(sigma^{mu nu}) = c sigma^{mu nu} with c =", c_S, "; dual(sigmabar^{mu nu}) = c sigmabar^{mu nu} with c =", c_Sb)
print("[PASS] sigma^{mu nu} and sigmabar^{mu nu} are eigenvectors of the Hodge dual with opposite eigenvalues -+ i:", len(c_S) == 1 and len(c_Sb) == 1 and c_S[0] == -c_Sb[0])
span = matrix(QQbar, [S[m][n].list() for m in range(4) for n in range(4)]).rank()
print("[PASS] the six sigma^{mu nu} span a three-dimensional space (the traceless 2x2 matrices):", span == 3 and all(S[m][n].trace() == 0 for m in range(4) for n in range(4)))

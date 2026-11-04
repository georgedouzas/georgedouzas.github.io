# Exercise exr-gth-dirac-gauge: the Dirac Lagrangian psibar (i gamma^mu D_mu - m) psi with D_mu = d_mu + i q A_mu in
# 1+1 dimensions (signature (+, -), gamma^0 = sigma_1, gamma^1 = i sigma_2) is invariant under psi -> e^{i q alpha} psi,
# A_mu -> A_mu - d_mu alpha, and its coupling to A_mu is -q psibar gamma^mu psi A_mu.
t, x, m, q = var("t x m q", domain="real")
a1, b1, a2, b2, al, A0, A1 = [function(n)(t, x) for n in ["a1", "b1", "a2", "b2", "al", "A0", "A1"]]
X = [t, x]; eta = diagonal_matrix([1, -1])
s1 = matrix(SR, [[0, 1], [1, 0]]); s2 = matrix(SR, [[0, -I], [I, 0]])
gam = [s1, I * s2]
print("[PASS] Clifford algebra {gamma^mu, gamma^nu} = 2 eta^{mu nu}:", all(gam[a] * gam[b] + gam[b] * gam[a] == 2 * eta[a, b] * identity_matrix(2) for a in range(2) for b in range(2)))
psi = vector(SR, [a1 + I * b1, a2 + I * b2]); psic = vector(SR, [a1 - I * b1, a2 - I * b2])
def L(ps, psc, A):
    psibar = psc * gam[0]
    Dpsi = [diff(ps, X[k]) + I * q * A[k] * ps for k in range(2)]
    return psibar * (sum((I * gam[k] * Dpsi[k] for k in range(2)), vector(SR, [0, 0])) - m * ps)
A = [A0, A1]
Ag = [A0 - diff(al, t), A1 - diff(al, x)]
dL = (L(exp(I * q * al) * psi, exp(-I * q * al) * psic, Ag) - L(psi, psic, A)).simplify_full()
print("[PASS] the gauged Dirac Lagrangian is gauge invariant:", bool(dL == 0))
c0, c1 = var("c0 c1", domain="real")
Lc = L(psi, psic, [c0, c1]).expand()
coup = [diff(Lc, c) for c in (c0, c1)]
cur = [(psic * gam[0]) * (gam[k] * psi) for k in range(2)]
print("coefficient of A_0, A_1:", [cc.simplify_full() for cc in coup])
print("[PASS] the coupling is -q psibar gamma^mu psi A_mu:", all(bool((coup[k] + q * cur[k]).simplify_full() == 0) for k in range(2)))
print("psibar gamma^0 psi = psi^dagger psi =", cur[0].expand(), "(a positive density)")

# Exercise exr-wz-closure: the commutator of two supersymmetry transformations on the scalar of the Wess-Zumino
# model is a translation, (delta_2 delta_1 - delta_1 delta_2) phi = i(-eps_1 sigma^mu eps_2^dag + eps_2 sigma^mu
# eps_1^dag) d_mu phi (Martin eq. 3.1.9), and the same holds on F (eq. 3.1.16) without using equations of motion.
# Grassmann parameters and fields are generators of an exterior algebra over the ring of commuting jets.
K0 = QuadraticField(-1, "I"); I = K0.gen(); MU = range(4)
bnames = ["phi", "phis", "F", "Fs"] + ["dphi%d" % m for m in MU] + ["dphis%d" % m for m in MU] + ["dF%d" % m for m in MU] + ["dFs%d" % m for m in MU] \
         + ["ddphi%d%d" % (m, n) for m in MU for n in MU if m <= n] + ["ddphis%d%d" % (m, n) for m in MU for n in MU if m <= n]
R = PolynomialRing(K0, bnames); B = dict(zip(bnames, R.gens()))
fnames = ["psi%d" % a for a in (0, 1)] + ["psis%d" % a for a in (0, 1)] + ["dpsi%d_%d" % (m, a) for m in MU for a in (0, 1)] \
         + ["dpsis%d_%d" % (m, a) for m in MU for a in (0, 1)] + ["ddpsi%d%d_%d" % (m, n, a) for m in MU for n in MU if m <= n for a in (0, 1)] \
         + ["ddpsis%d%d_%d" % (m, n, a) for m in MU for n in MU if m <= n for a in (0, 1)] \
         + ["e1_%d" % a for a in (0, 1)] + ["e1s_%d" % a for a in (0, 1)] + ["e2_%d" % a for a in (0, 1)] + ["e2s_%d" % a for a in (0, 1)]
E = ExteriorAlgebra(R, fnames); G = dict(zip(fnames, E.gens())); gens = list(E.gens()); idx = {str(g): i for i, g in enumerate(gens)}
def sym(m, n): return (m, n) if m <= n else (n, m)
phi, phis, F, Fs = B["phi"], B["phis"], B["F"], B["Fs"]
dphi = [B["dphi%d" % m] for m in MU]; dphis = [B["dphis%d" % m] for m in MU]; dF = [B["dF%d" % m] for m in MU]; dFs = [B["dFs%d" % m] for m in MU]
psi = [G["psi%d" % a] for a in (0, 1)]; psis = [G["psis%d" % a] for a in (0, 1)]
dpsi = [[G["dpsi%d_%d" % (m, a)] for a in (0, 1)] for m in MU]; dpsis = [[G["dpsis%d_%d" % (m, a)] for a in (0, 1)] for m in MU]
e1 = [G["e1_%d" % a] for a in (0, 1)]; e1s = [G["e1s_%d" % a] for a in (0, 1)]; e2 = [G["e2_%d" % a] for a in (0, 1)]; e2s = [G["e2s_%d" % a] for a in (0, 1)]
pauli = [matrix(K0, [[0, 1], [1, 0]]), matrix(K0, [[0, -I], [I, 0]]), matrix(K0, [[1, 0], [0, -1]])]
sig = [identity_matrix(K0, 2)] + pauli; sigb = [identity_matrix(K0, 2)] + [-p for p in pauli]
def up(x): return [x[1], -x[0]]
def dot(x, y): return sum(up(x)[a] * y[a] for a in (0, 1))
def dotbar(x, y): return sum(x[a] * up(y)[a] for a in (0, 1))
def xbar_s_y(x, S, y): return sum(x[a] * S[a, b] * y[b] for a in (0, 1) for b in (0, 1))
def x_s_ybar(x, S, y): return sum(up(x)[a] * S[a, b] * up(y)[b] for a in (0, 1) for b in (0, 1))
def derivation(expr, bos, ferm):
    out = E.zero()
    for key, c in expr.monomial_coefficients().items():
        ii = sorted(key); mono = prod((gens[i] for i in ii), E.one())
        for v, Dv in bos.items():
            dc = c.derivative(v)
            if dc != 0: out += dc * Dv * mono
        for j in range(len(ii)):
            pre = prod((gens[i] for i in ii[:j]), E.one()); post = prod((gens[i] for i in ii[j + 1:]), E.one())
            out += c * pre * ferm[ii[j]] * post
    return out
def D(mu):
    bos = {phi: E(dphi[mu]), phis: E(dphis[mu]), F: E(dF[mu]), Fs: E(dFs[mu])}
    for n in MU: bos[dphi[n]] = E(B["ddphi%d%d" % sym(mu, n)]); bos[dphis[n]] = E(B["ddphis%d%d" % sym(mu, n)])
    ferm = {}
    for a in (0, 1):
        ferm[idx[str(psi[a])]] = dpsi[mu][a]; ferm[idx[str(psis[a])]] = dpsis[mu][a]
        for n in MU:
            ferm[idx[str(dpsi[n][a])]] = G["ddpsi%d%d_%d" % (sym(mu, n) + (a,))]; ferm[idx[str(dpsis[n][a])]] = G["ddpsis%d%d_%d" % (sym(mu, n) + (a,))]
        for e in (e1, e1s, e2, e2s): ferm[idx[str(e[a])]] = E.zero()
    return lambda x: derivation(x, bos, ferm)
def make_delta(e, es):
    d_phi = dot(e, psi); d_phis = dotbar(es, psis)
    d_psi = [sum(-I * sig[m][a, b] * up(es)[b] * dphi[m] for m in MU for b in (0, 1)) + e[a] * F for a in (0, 1)]
    d_psis = [sum(I * up(e)[b] * sig[m][b, a] * dphis[m] for m in MU for b in (0, 1)) + es[a] * Fs for a in (0, 1)]
    d_F = -I * sum(xbar_s_y(es, sigb[m], dpsi[m]) for m in MU); d_Fs = I * sum(xbar_s_y(dpsis[m], sigb[m], e) for m in MU)
    Dm = [D(m) for m in MU]
    bos = {phi: d_phi, phis: d_phis, F: d_F, Fs: d_Fs}
    for m in MU: bos[dphi[m]] = Dm[m](d_phi); bos[dphis[m]] = Dm[m](d_phis); bos[dF[m]] = Dm[m](d_F); bos[dFs[m]] = Dm[m](d_Fs)
    ferm = {}
    for a in (0, 1):
        ferm[idx[str(psi[a])]] = d_psi[a]; ferm[idx[str(psis[a])]] = d_psis[a]
        for m in MU: ferm[idx[str(dpsi[m][a])]] = Dm[m](d_psi[a]); ferm[idx[str(dpsis[m][a])]] = Dm[m](d_psis[a])
        for ee in (e1, e1s, e2, e2s): ferm[idx[str(ee[a])]] = E.zero()
    return lambda x: derivation(x, bos, ferm)
d1 = make_delta(e1, e1s); d2 = make_delta(e2, e2s)
xi = [I * (-x_s_ybar(e1, sig[m], e2s) + x_s_ybar(e2, sig[m], e1s)) for m in MU]   # the translation parameter, eq. 3.1.9
comm_phi = d2(d1(E(phi))) - d1(d2(E(phi)))
print("[PASS] (delta_2 delta_1 - delta_1 delta_2) phi = i(-eps_1 sigma^mu eps_2^dag + eps_2 sigma^mu eps_1^dag) d_mu phi:",
      comm_phi - sum(xi[m] * dphi[m] for m in MU) == 0)
comm_F = d2(d1(E(F))) - d1(d2(E(F)))
print("[PASS] the same commutator on the auxiliary field F, off-shell (eq. 3.1.16):", comm_F - sum(xi[m] * dF[m] for m in MU) == 0)
comm_psi = [d2(d1(psi[a])) - d1(d2(psi[a])) for a in (0, 1)]
print("[PASS] the same commutator on psi_alpha, off-shell thanks to F (eq. 3.1.16):", all(comm_psi[a] - sum(xi[m] * dpsi[m][a] for m in MU) == 0 for a in (0, 1)))

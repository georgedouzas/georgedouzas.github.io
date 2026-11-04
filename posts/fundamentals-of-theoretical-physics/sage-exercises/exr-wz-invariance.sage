# Exercise exr-wz-invariance: the Wess-Zumino model is supersymmetric off-shell. Fields: a complex scalar phi, a
# left-handed Weyl spinor psi_alpha and the auxiliary field F, with Martin's conventions (hep-ph/9709356, Sec. 3.1)
# converted to our signature (+, -, -, -): the scalar term of L and the d^mu terms of K^mu change sign relative to
# Martin's mostly-plus metric, the fermion and auxiliary terms do not. The anticommuting fields and their
# derivatives are generators of an exterior algebra over
# the polynomial ring of the commuting jets, so every Grassmann sign is handled by Sage; a supersymmetry
# transformation delta is an even derivation of this algebra and d_mu is another. We compute delta L and check
# that it equals d_mu K^mu for Martin's K^mu (eq. 3.1.7) plus the auxiliary-field term, as an exact identity.
K0 = QuadraticField(-1, "I"); I = K0.gen()
MU = range(4); eta = [1, -1, -1, -1]
# commuting jets: phi, phi*, their first and symmetric second derivatives, F, F* and first derivatives
bnames = (["phi", "phis", "F", "Fs"] + ["dphi%d" % m for m in MU] + ["dphis%d" % m for m in MU]
          + ["ddphi%d%d" % (m, n) for m in MU for n in MU if m <= n] + ["ddphis%d%d" % (m, n) for m in MU for n in MU if m <= n]
          + ["dF%d" % m for m in MU] + ["dFs%d" % m for m in MU])
R = PolynomialRing(K0, bnames); B = dict(zip(bnames, R.gens()))
# anticommuting jets and parameters
fnames = (["psi%d" % a for a in (0, 1)] + ["psis%d" % a for a in (0, 1)]
          + ["dpsi%d_%d" % (m, a) for m in MU for a in (0, 1)] + ["dpsis%d_%d" % (m, a) for m in MU for a in (0, 1)]
          + ["ddpsi%d%d_%d" % (m, n, a) for m in MU for n in MU if m <= n for a in (0, 1)]
          + ["ddpsis%d%d_%d" % (m, n, a) for m in MU for n in MU if m <= n for a in (0, 1)]
          + ["eps%d" % a for a in (0, 1)] + ["epss%d" % a for a in (0, 1)])
E = ExteriorAlgebra(R, fnames); G = dict(zip(fnames, E.gens())); gens = list(E.gens()); idx = {str(g): i for i, g in enumerate(gens)}
def sym(m, n): return (m, n) if m <= n else (n, m)
phi, phis, F, Fs = B["phi"], B["phis"], B["F"], B["Fs"]
dphi = [B["dphi%d" % m] for m in MU]; dphis = [B["dphis%d" % m] for m in MU]
dF = [B["dF%d" % m] for m in MU]; dFs = [B["dFs%d" % m] for m in MU]
psi = [G["psi%d" % a] for a in (0, 1)]; psis = [G["psis%d" % a] for a in (0, 1)]
dpsi = [[G["dpsi%d_%d" % (m, a)] for a in (0, 1)] for m in MU]; dpsis = [[G["dpsis%d_%d" % (m, a)] for a in (0, 1)] for m in MU]
eps = [G["eps%d" % a] for a in (0, 1)]; epss = [G["epss%d" % a] for a in (0, 1)]
# sigma matrices (sigma^mu)_{alpha alphadot} and (sigmabar^mu)^{alphadot alpha}, Martin eq. 2.7
pauli = [matrix(K0, [[0, 1], [1, 0]]), matrix(K0, [[0, -I], [I, 0]]), matrix(K0, [[1, 0], [0, -1]])]
sig = [identity_matrix(K0, 2)] + pauli; sigb = [identity_matrix(K0, 2)] + [-p for p in pauli]
def up(x): return [x[1], -x[0]]                       # xi^1 = eps^{12} xi_2, xi^2 = eps^{21} xi_1
def dot(x, y): return sum(up(x)[a] * y[a] for a in (0, 1))              # xi chi = xi^a chi_a
def dotbar(x, y): return sum(x[a] * up(y)[a] for a in (0, 1))           # xi^dag chi^dag = xi^dag_adot chi^dag^adot
def xbar_s_y(x, S, y): return sum(x[a] * S[a, b] * y[b] for a in (0, 1) for b in (0, 1))        # x^dag sigmabar y
def x_s_ybar(x, S, y): return sum(up(x)[a] * S[a, b] * up(y)[b] for a in (0, 1) for b in (0, 1))  # x sigma y^dag
# the Lagrangian, Martin eqs. 3.1.2 and 3.1.12 with the scalar term d^mu phi* d_mu phi in our signature
L = sum(eta[m] * dphis[m] * dphi[m] for m in MU) + I * sum(xbar_s_y(psis, sigb[m], dpsi[m]) for m in MU) + Fs * F
# an even derivation of E from its values on the commuting variables (dict var -> E) and on the generators
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
# the spacetime derivative d_mu as a derivation
def D(mu):
    bos = {phi: E(dphi[mu]), phis: E(dphis[mu]), F: E(dF[mu]), Fs: E(dFs[mu])}
    for n in MU:
        bos[dphi[n]] = E(B["ddphi%d%d" % sym(mu, n)]); bos[dphis[n]] = E(B["ddphis%d%d" % sym(mu, n)])
    ferm = {}
    for a in (0, 1):
        ferm[idx[str(psi[a])]] = dpsi[mu][a]; ferm[idx[str(psis[a])]] = dpsis[mu][a]
        for n in MU:
            ferm[idx[str(dpsi[n][a])]] = G["ddpsi%d%d_%d" % (sym(mu, n) + (a,))]
            ferm[idx[str(dpsis[n][a])]] = G["ddpsis%d%d_%d" % (sym(mu, n) + (a,))]
        ferm[idx[str(eps[a])]] = E.zero(); ferm[idx[str(epss[a])]] = E.zero()   # constant parameter
    return lambda x: derivation(x, bos, ferm)
# the supersymmetry transformation, Martin eqs. 3.1.3, 3.1.13, 3.1.15, extended to the derivative jets by d(delta X)
def make_delta(e, es, withF=True):
    d_phi = dot(e, psi); d_phis = dotbar(es, psis)
    d_psi = [sum(-I * sig[m][a, b] * up(es)[b] * dphi[m] for m in MU for b in (0, 1)) + (e[a] * F if withF else 0) for a in (0, 1)]
    d_psis = [sum(I * up(e)[b] * sig[m][b, a] * dphis[m] for m in MU for b in (0, 1)) + (es[a] * Fs if withF else 0) for a in (0, 1)]
    d_F = -I * sum(xbar_s_y(es, sigb[m], dpsi[m]) for m in MU)
    d_Fs = I * sum(xbar_s_y(dpsis[m], sigb[m], e) for m in MU)
    Dm = [D(m) for m in MU]
    bos = {phi: d_phi, phis: d_phis, F: d_F, Fs: d_Fs}
    for m in MU:
        bos[dphi[m]] = Dm[m](d_phi); bos[dphis[m]] = Dm[m](d_phis); bos[dF[m]] = Dm[m](d_F); bos[dFs[m]] = Dm[m](d_Fs)
    ferm = {}
    for a in (0, 1):
        ferm[idx[str(psi[a])]] = d_psi[a]; ferm[idx[str(psis[a])]] = d_psis[a]
        for m in MU:
            ferm[idx[str(dpsi[m][a])]] = Dm[m](d_psi[a]); ferm[idx[str(dpsis[m][a])]] = Dm[m](d_psis[a])
        ferm[idx[str(e[a])]] = E.zero(); ferm[idx[str(es[a])]] = E.zero()
    return lambda x: derivation(x, bos, ferm)
delta = make_delta(eps, epss)
dL = delta(L)
print("number of terms in delta L:", len(dL.monomial_coefficients()))
# K^mu from Martin eq. 3.1.7 (the total-derivative part of delta L_fermion) plus the auxiliary-field term, converted:
# K^mu = -eps sigma^nu sigmabar^mu psi d_nu phi* + eps psi d^mu phi* + eps^dag psi^dag d^mu phi + i psi^dag sigmabar^mu eps F
def Kmu(mu):
    k = (-sum(up(eps)[a] * sig[n][a, b] * sigb[mu][b, c] * psi[c] * dphis[n] for n in MU for a in (0, 1) for b in (0, 1) for c in (0, 1))
         + dot(eps, psi) * eta[mu] * dphis[mu] + dotbar(epss, psis) * eta[mu] * dphi[mu])
    return k + I * xbar_s_y(psis, sigb[mu], eps) * F
divK = sum(D(m)(Kmu(m)) for m in MU)
print("[PASS] delta L = d_mu K^mu exactly (off-shell, with the auxiliary field):", dL - divK == 0)
# without the auxiliary field (Martin eqs. 3.1.3-3.1.8): delta_0 psi = -i sigma^mu eps^dag d_mu phi only, and
# delta_0 L_0 = d_mu K_0^mu with K_0 the F-independent part of K
L0 = sum(eta[m] * dphis[m] * dphi[m] for m in MU) + I * sum(xbar_s_y(psis, sigb[m], dpsi[m]) for m in MU)
delta0 = make_delta(eps, epss, withF=False)
K0 = lambda mu: Kmu(mu) - I * xbar_s_y(psis, sigb[mu], eps) * F
print("[PASS] the massless free model without F: delta_0 L_0 = d_mu K_0^mu (Martin eq. 3.1.8):", delta0(L0) - sum(D(m)(K0(m)) for m in MU) == 0)

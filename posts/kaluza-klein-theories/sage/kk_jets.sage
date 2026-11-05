# Oracle: the abelian Kaluza-Klein reduction as a polynomial identity in jet variables (design D3). The ansatz
# ds^2 = g_{mu nu} dx dx - phi^2 (dy + A_mu dx^mu)^2 with g, A, phi independent of y (signature (+,-,-,-,-): the circle
# direction is spacelike, so the internal block carries a minus sign). The jets (values, first and second
# derivatives of g, A, phi at a point, and symbols ginv for the inverse metric) generate a polynomial ring; the
# five-dimensional Christoffel symbols, Ricci tensor and curvature scalar are the standard formulas evaluated on these
# symbols, with d_rho acting as a derivation that sends ginv to -ginv dg ginv. The oracle checks that
#   sqrt(-g5) = phi sqrt(-g4) (as det g5 = -phi^2 det g4),
#   Gamma5 patterns, Ric5 components and R5 = R4 + (phi^2/4) F^2 - (2/phi) Box phi, Box = g^{mu nu} nabla_mu d_nu
# hold as exact polynomial identities, printing the sizes that fix the Lean strategy. DIAG=1 uses a diagonal 4D metric.
import os, sys
DIAG = os.environ.get("KK_DIAG", "0") == "1"
MU = range(4); FIVE = range(5)
def sym(a, b): return (a, b) if a <= b else (b, a)
names = []
pairs = [(m, n) for m in MU for n in MU if m <= n]
if DIAG:
    names += ["g%d" % m for m in MU] + ["dg%d_%d" % (r, m) for r in MU for m in MU] + ["ddg%d%d_%d" % (r, s, m) for (r, s) in pairs for m in MU]
else:
    names += ["g%d%d" % p for p in pairs] + ["ginv%d%d" % p for p in pairs]
    names += ["dg%d_%d%d" % ((r,) + p) for r in MU for p in pairs] + ["ddg%d%d_%d%d" % (rs + p) for rs in pairs for p in pairs]
names += ["A%d" % m for m in MU] + ["dA%d_%d" % (r, m) for r in MU for m in MU] + ["ddA%d%d_%d" % (rs + (m,)) for rs in pairs for m in MU]
names += ["phi", "iphi"] + ["dphi%d" % m for m in MU] + ["ddphi%d%d" % rs for rs in pairs]
if DIAG: names += ["ig%d" % m for m in MU]
R = PolynomialRing(QQ, names); V = dict(zip(names, R.gens()))
def g(m, n):
    if DIAG: return V["g%d" % m] if m == n else R(0)
    return V["g%d%d" % sym(m, n)]
def ginv(m, n):
    if DIAG: return V["ig%d" % m] if m == n else R(0)
    return V["ginv%d%d" % sym(m, n)]
def dg(r, m, n):
    if DIAG: return V["dg%d_%d" % (r, m)] if m == n else R(0)
    return V["dg%d_%d%d" % ((r,) + sym(m, n))]
def ddg(r, s, m, n):
    if DIAG: return V["ddg%d%d_%d" % (sym(r, s) + (m,))] if m == n else R(0)
    return V["ddg%d%d_%d%d" % (sym(r, s) + sym(m, n))]
A = [V["A%d" % m] for m in MU]; dA = lambda r, m: V["dA%d_%d" % (r, m)]; ddA = lambda r, s, m: V["ddA%d%d_%d" % (sym(r, s) + (m,))]
phi, iphi = V["phi"], V["iphi"]; dphi = [V["dphi%d" % m] for m in MU]; ddphi = lambda r, s: V["ddphi%d%d" % sym(r, s)]
# the derivation d_rho on the jet ring (first-order jets only; second derivatives of ginv are never needed)
def D(r, p):
    out = R(0)
    for v in p.variables():
        n = str(v)
        if n == "phi": dv = dphi[r]
        elif n == "iphi": dv = -iphi^2 * dphi[r]
        elif n.startswith("dphi"): dv = ddphi(r, int(n[4]))
        elif n.startswith("ddphi"): raise ValueError("third derivative of phi")
        elif n.startswith("A"): dv = dA(r, int(n[1]))
        elif n.startswith("dA"): m = int(n.split("_")[1]); dv = ddA(r, int(n[2]), m)
        elif n.startswith("ginv"): a, b = int(n[4]), int(n[5]); dv = -sum(ginv(a, c) * dg(r, c, d) * ginv(d, b) for c in MU for d in MU)
        elif n.startswith("ig"): m = int(n[2]); dv = -V["ig%d" % m]^2 * dg(r, m, m)
        elif n.startswith("g"):
            if DIAG: m = int(n[1]); dv = dg(r, m, m)
            else: a, b = int(n[1]), int(n[2]); dv = dg(r, a, b)
        elif n.startswith("dg"):
            if DIAG: rr, m = int(n[2]), int(n.split("_")[1]); dv = ddg(r, rr, m, m)
            else: rr = int(n[2]); a, b = int(n.split("_")[1][0]), int(n.split("_")[1][1]); dv = ddg(r, rr, a, b)
        elif n.startswith("ddg") or n.startswith("ddA"): raise ValueError("third derivative")
        else: raise ValueError(n)
        out += p.derivative(v) * dv
    return out
# five-dimensional metric jets (index 4 is y); the cylinder condition sets every y-derivative to zero
def G(i, j):
    if i < 4 and j < 4: return g(i, j) - phi^2 * A[i] * A[j]
    if i < 4: return -phi^2 * A[i]
    if j < 4: return -phi^2 * A[j]
    return -phi^2
def Ginv(i, j):
    if i < 4 and j < 4: return ginv(i, j)
    if i < 4: return -sum(ginv(i, k) * A[k] for k in MU)
    if j < 4: return -sum(ginv(j, k) * A[k] for k in MU)
    return -iphi^2 + sum(ginv(k, l) * A[k] * A[l] for k in MU for l in MU)
def dG(r, i, j): return R(0) if r == 4 else D(r, G(i, j))
# checks of the inverse (needs ginv g = 1 only in the generic case: substitute later) and the determinant
Gm = matrix(R, 5, 5, lambda i, j: G(i, j)); Gi = matrix(R, 5, 5, lambda i, j: Ginv(i, j))
gm = matrix(R, 4, 4, lambda i, j: g(i, j))
print("[PASS] det g5 = -phi^2 det g4 as a polynomial identity:", Gm.det() == -phi^2 * gm.det())
if DIAG:
    prod_check = (Gi * Gm).subs({V["ig%d" % m]: 1 / V["g%d" % m] for m in MU}).subs({iphi: 1 / phi})
    print("[PASS] the block inverse of the ansatz metric is the inverse (diagonal 4D metric):", prod_check == identity_matrix(5))
Fr = R.fraction_field()
def Z(p): return Fr(p)(iphi=1 / phi) == 0        # zero after substituting the inverse of phi (the only relation used)
Gam5 = {(k, i, j): sum(Ginv(k, l) * (dG(i, l, j) + dG(j, l, i) - dG(l, i, j)) for l in FIVE) / 2 for k in FIVE for i in FIVE for j in FIVE}
Gam4 = {(k, i, j): sum(ginv(k, l) * (dg(i, l, j) + dg(j, l, i) - dg(l, i, j)) for l in MU) / 2 for k in MU for i in MU for j in MU}
F = lambda m, n: dA(m, n) - dA(n, m)
Fup = lambda m, k: sum(ginv(k, l) * F(m, l) for l in MU)                      # F_m^k
Aup = lambda k: sum(ginv(k, l) * A[l] for l in MU)
dphiup = lambda k: sum(ginv(k, l) * dphi[l] for l in MU)
# Christoffel patterns (Pope's lectures with phi^2 -> -phi^2 for the mostly-minus signature), checked as polynomial identities
ok1 = all(Z(Gam5[(k, m, n)] - (Gam4[(k, m, n)] - phi^2 / 2 * (A[m] * Fup(n, k) + A[n] * Fup(m, k)) + phi * A[m] * A[n] * dphiup(k))) for k in MU for m in MU for n in MU)
ok2 = all(Z(Gam5[(k, m, 4)] - (-phi^2 / 2 * Fup(m, k) + phi * A[m] * dphiup(k))) for k in MU for m in MU)
ok3 = all(Z(Gam5[(k, 4, 4)] - phi * dphiup(k)) for k in MU)
ok4 = all(Z(Gam5[(4, m, n)] - (iphi * (A[m] * dphi[n] + A[n] * dphi[m]) + (dA(m, n) + dA(n, m)) / 2 - sum(A[k] * Gam5[(k, m, n)] for k in MU))) for m in MU for n in MU)
ok5 = all(Z(Gam5[(4, m, 4)] - (iphi * dphi[m] - sum(A[k] * Gam5[(k, m, 4)] for k in MU))) for m in MU)
ok6 = Z(Gam5[(4, 4, 4)] + sum(A[k] * Gam5[(k, 4, 4)] for k in MU))
print("[PASS] the six Christoffel patterns of the ansatz hold as polynomial identities in the jets:", ok1 and ok2 and ok3 and ok4 and ok5 and ok6)
# Ricci tensors from the standard formula with d_k Gamma computed by the derivation
def ricci(Gam, inv, dim, Dfun):
    idx = range(dim)
    return {(i, j): sum(Dfun(k, Gam[(k, i, j)]) for k in idx) - sum(Dfun(j, Gam[(k, i, k)]) for k in idx)
            + sum(Gam[(k, k, l)] * Gam[(l, i, j)] for k in idx for l in idx) - sum(Gam[(k, j, l)] * Gam[(l, i, k)] for k in idx for l in idx) for i in idx for j in idx}
D5 = lambda r, p: R(0) if r == 4 else D(r, p)
Ric5 = ricci(Gam5, Ginv, 5, D5); Ric4 = ricci(Gam4, ginv, 4, D)
R5 = sum(Ginv(i, j) * Ric5[(i, j)] for i in FIVE for j in FIVE); R4 = sum(ginv(i, j) * Ric4[(i, j)] for i in MU for j in MU)
F2 = sum(ginv(m, k) * ginv(n, l) * F(m, n) * F(k, l) for m in MU for n in MU for k in MU for l in MU)
box = sum(ginv(m, n) * (ddphi(m, n) - sum(Gam4[(k, m, n)] * dphi[k] for k in MU)) for m in MU for n in MU)
diff = R5 - (R4 + phi^2 / 4 * F2 - 2 * iphi * box)
print("[PASS] R5 = R4 + (phi^2/4) F^2 - (2/phi) Box phi as a polynomial identity in the jets (ginv free, %s 4D metric):" % ("diagonal" if DIAG else "generic"), Z(diff))
print("sizes: monomials in R5 =", len(R5.monomials()), ", in R4 =", len(R4.monomials()), ", in Ric5_00 =", len(Ric5[(0, 0)].monomials()), ", Ric5_04 =", len(Ric5[(0, 4)].monomials()), ", Ric5_44 =", len(Ric5[(4, 4)].monomials()))
# Ricci components (R_{mu nu}, R_{mu 4}, R_{44}) in closed form
nabla2 = lambda m, n: ddphi(m, n) - sum(Gam4[(k, m, n)] * dphi[k] for k in MU)
FF = lambda m, n: sum(ginv(k, l) * F(m, k) * F(n, l) for k in MU for l in MU)               # F_m^k F_nk
divF = lambda n: sum(ginv(m, k) * (D(m, F(k, n)) - sum(Gam4[(l, m, k)] * F(l, n) + Gam4[(l, m, n)] * F(k, l) for l in MU)) for m in MU for k in MU)  # nabla^m F_{m n}
R44 = Ric5[(4, 4)] - (phi * box + phi^4 / 4 * F2)
Rm4 = {m: Ric5[(m, 4)] - (A[m] * Ric5[(4, 4)] + phi^2 / 2 * divF(m) - 3 * phi / 2 * sum(dphi[k] * Fup(m, k) for k in MU)) for m in MU}
Rmn = {(m, n): Ric5[(m, n)] - (Ric4[(m, n)] + phi^2 / 2 * FF(m, n) - iphi * nabla2(m, n) + A[m] * Ric5[(n, 4)] + A[n] * Ric5[(m, 4)] - A[m] * A[n] * Ric5[(4, 4)]) for m in MU for n in MU}
print("[PASS] R_44 = phi Box phi + (phi^4/4) F^2:", Z(R44))
print("[PASS] R_{mu 4} = A_mu R_44 + (phi^2/2) nabla^k F_{k mu} - (3 phi/2) d_k phi F_mu^k:", all(Z(v) for v in Rm4.values()))
print("[PASS] R_{mu nu} = R4_{mu nu} + (phi^2/2) F_mu^k F_nu k - (1/phi) nabla_mu d_nu phi + A_mu R_{nu 4} + A_nu R_{mu 4} - A_mu A_nu R_44:", all(Z(v) for v in Rmn.values()))

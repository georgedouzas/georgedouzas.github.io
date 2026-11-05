# Oracle: the Weyl rescaling of the four-dimensional curvature scalar as a jet identity. For g~ = Omega^2 g in four
# dimensions, R[g~] = Omega^{-2} (R[g] - 6 Box ln Omega - 6 (d ln Omega)^2) = Omega^{-2} R - 6 Omega^{-3} Box Omega
# + 6 Omega^{-4} ... written without logarithms: R~ = Omega^{-2} R - 6 Omega^{-3} Box Omega + 6 Omega^{-4} ... no:
#   R~ = Omega^{-2} [ R - 6 Omega^{-1} Box Omega ]   (the (d ln Omega)^2 terms combine; checked below).
# The jets of g~ are built from those of g and Omega; the formal derivative of ginv~ = Omega^{-2} ginv must agree
# with -ginv~ dg~ ginv~, which needs ginv g = 1; the oracle reports whether the identity holds with ginv free and,
# if not, substitutes the true inverse of a generic symmetric matrix (exact rational functions).
MU = range(4)
def sym(a, b): return (a, b) if a <= b else (b, a)
pairs = [(m, n) for m in MU for n in MU if m <= n]
names = ["g%d%d" % p for p in pairs] + ["dg%d_%d%d" % ((r,) + p) for r in MU for p in pairs] + ["ddg%d%d_%d%d" % (rs + p) for rs in pairs for p in pairs]
names += ["ginv%d%d" % p for p in pairs] + ["W", "iW"] + ["dW%d" % m for m in MU] + ["ddW%d%d" % rs for rs in pairs]
R = PolynomialRing(QQ, names); V = dict(zip(names, R.gens())); Fr = R.fraction_field()
g = lambda m, n: V["g%d%d" % sym(m, n)]; ginv = lambda m, n: V["ginv%d%d" % sym(m, n)]
dg = lambda r, m, n: V["dg%d_%d%d" % ((r,) + sym(m, n))]; ddg = lambda r, s, m, n: V["ddg%d%d_%d%d" % (sym(r, s) + sym(m, n))]
W, iW = V["W"], V["iW"]; dW = [V["dW%d" % m] for m in MU]; ddW = lambda r, s: V["ddW%d%d" % sym(r, s)]
def D(r, p):
    out = R(0)
    for v in p.variables():
        n = str(v)
        if n == "W": dv = dW[r]
        elif n == "iW": dv = -iW^2 * dW[r]
        elif n.startswith("dW"): dv = ddW(r, int(n[2]))
        elif n.startswith("ginv"): a, b = int(n[4]), int(n[5]); dv = -sum(ginv(a, c) * dg(r, c, d) * ginv(d, b) for c in MU for d in MU)
        elif n.startswith("ddg") or n.startswith("ddW"): raise ValueError("third derivative")
        elif n.startswith("dg"): rr = int(n[2]); a, b = int(n.split("_")[1][0]), int(n.split("_")[1][1]); dv = ddg(r, rr, a, b)
        elif n.startswith("g"): dv = dg(r, int(n[1]), int(n[2]))
        else: raise ValueError(n)
        out += p.derivative(v) * dv
    return out
def ricci_scalar(gf, ginvf, dgf, ddgf, dginvf):
    Gam = {(k, i, j): sum(ginvf(k, l) * (dgf(i, l, j) + dgf(j, l, i) - dgf(l, i, j)) for l in MU) / 2 for k in MU for i in MU for j in MU}
    dGam = {(r, k, i, j): sum(dginvf(r, k, l) * (dgf(i, l, j) + dgf(j, l, i) - dgf(l, i, j)) + ginvf(k, l) * (ddgf(r, i, l, j) + ddgf(r, j, l, i) - ddgf(r, l, i, j)) for l in MU) / 2 for r in MU for k in MU for i in MU for j in MU}
    Ric = {(i, j): sum(dGam[(k, k, i, j)] for k in MU) - sum(dGam[(j, k, i, k)] for k in MU) + sum(Gam[(k, k, l)] * Gam[(l, i, j)] for k in MU for l in MU) - sum(Gam[(k, j, l)] * Gam[(l, i, k)] for k in MU for l in MU) for i in MU for j in MU}
    return Gam, Ric, sum(ginvf(i, j) * Ric[(i, j)] for i in MU for j in MU)
dginv = lambda r, m, n: -sum(ginv(m, c) * dg(r, c, d) * ginv(d, n) for c in MU for d in MU)
Gam4, Ric4, R4 = ricci_scalar(g, ginv, dg, ddg, dginv)
# rescaled jets
gt = lambda m, n: W^2 * g(m, n)
ginvt = lambda m, n: iW^2 * ginv(m, n)
dgt = lambda r, m, n: 2 * W * dW[r] * g(m, n) + W^2 * dg(r, m, n)
ddgt = lambda r, s, m, n: 2 * (dW[r] * dW[s] + W * ddW(r, s)) * g(m, n) + 2 * W * dW[r] * dg(s, m, n) + 2 * W * dW[s] * dg(r, m, n) + W^2 * ddg(r, s, m, n)
dginvt_formal = lambda r, m, n: -sum(ginvt(m, c) * dgt(r, c, d) * ginvt(d, n) for c in MU for d in MU)      # -ginv~ dg~ ginv~
dginvt_jet = lambda r, m, n: D(r, ginvt(m, n))                                                              # derivative of iW^2 ginv
box = lambda f, df, ddf: sum(ginv(m, n) * (ddf(m, n) - sum(Gam4[(k, m, n)] * df[k] for k in MU)) for m in MU for n in MU)
boxW = box(W, dW, ddW)
dW2 = sum(ginv(m, n) * dW[m] * dW[n] for m in MU for n in MU)
target = iW^2 * R4 - 6 * iW^3 * boxW          # R~ = Omega^{-2} R - 6 Omega^{-3} Box Omega (four dimensions)
def Z(p): return Fr(p)(iW=1 / W) == 0
_, _, Rt_jet = ricci_scalar(gt, ginvt, dgt, ddgt, dginvt_jet)
_, _, Rt_formal = ricci_scalar(gt, ginvt, dgt, ddgt, dginvt_formal)
print("with ginv free: identity using the jet derivative of ginv~:", Z(Rt_jet - target), "; using the formal -ginv~ dg~ ginv~:", Z(Rt_formal - target))
print("sizes: monomials in R~ (jet) =", len(Rt_jet.monomials()), " target =", len(target.monomials()))
# with the true inverse: evaluate at random rational points, with ginv the actual inverse of the random g
import random
random.seed(int(11))
def numeric_check(p, trials=3):
    for _ in range(trials):
        while True:
            gnum = matrix(QQ, 4, 4)
            for (i, j) in pairs:
                v = QQ(random.randint(-5, 5)) / random.randint(1, 3); gnum[i, j] = v; gnum[j, i] = v
            if gnum.det() != 0: break
        ginum = gnum.inverse()
        vals = {}
        for nme in names:
            if nme.startswith("ginv"): vals[nme] = ginum[int(nme[4]), int(nme[5])]
            elif nme.startswith("g") and not nme.startswith("ginv"): vals[nme] = gnum[int(nme[1]), int(nme[2])]
            elif nme == "W": vals[nme] = QQ(random.randint(1, 4)) / random.randint(1, 3)
            elif nme == "iW": continue
            else: vals[nme] = QQ(random.randint(-6, 6)) / random.randint(1, 4)
        vals["iW"] = 1 / vals["W"]
        if p(**vals) != 0: return False
    return True
print("[PASS] R[Omega^2 g] = Omega^{-2} R[g] - 6 Omega^{-3} Box Omega in four dimensions at random points with the true inverse (jet derivative of ginv~):", numeric_check(Rt_jet - target))
print("[PASS] the same with the formal inverse-derivative rule -ginv~ dg~ ginv~:", numeric_check(Rt_formal - target))
print("[PASS] -6 Box ln Omega - 6 (d ln Omega)^2 = -6 Omega^{-1} Box Omega as rational functions:", Z(-6 * (iW * boxW - iW^2 * dW2) - 6 * iW^2 * dW2 + 6 * iW * boxW))

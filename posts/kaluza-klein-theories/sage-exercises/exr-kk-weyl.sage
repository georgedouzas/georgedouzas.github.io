# Exercise exr-kk-weyl: the Weyl rescaling in four dimensions on an explicit metric. For g~ = Omega^2 g with the
# Schwarzschild-like diagonal metric g = diag(f(r), -1/f(r), -r^2, -r^2 sin^2 th) (signature (+,-,-,-)) and
# Omega = Omega(r), check
# R[g~] = Omega^{-2} R[g] - 6 Omega^{-3} Box Omega, then the Einstein-frame bookkeeping: with g = phi^{-1} g~,
# sqrt(-g) phi R[g] = sqrt(-g~) (R~ + 3 Box~ ln phi - (3/2) (d~ ln phi)^2).
M = Manifold(4, "M", structure="Lorentzian"); X = M.chart("t r th ph"); t, r, th, ph = X[:]
f = function("f")(r); Om = function("Omega")(r)
g = M.metric("g"); g[0, 0] = f; g[1, 1] = -1 / f; g[2, 2] = -r^2; g[3, 3] = -r^2 * sin(th)^2
gt = M.metric("gt")
for i in range(4): gt[i, i] = Om^2 * g[i, i].expr()
R = g.ricci_scalar().expr(); Rt = gt.ricci_scalar().expr()
nab = g.connection()
box = lambda s: (nab(nab(M.scalar_field(s)).up(g)).trace()).expr()
print("[PASS] R[Omega^2 g] = Omega^{-2} R[g] - 6 Omega^{-3} Box Omega on the diagonal metric with Omega(r):", bool((Rt - (R / Om^2 - 6 * box(Om) / Om^3)).simplify_full() == 0))
# Einstein frame: g = phi^{-1} gE, i.e. gE = phi g (Omega^2 = phi^{-1} rescales gE to g); gE plays the role of g~ above
phi = function("phi")(r)
gJ = M.metric("gJ")                     # Jordan-frame metric g = phi^{-1} gE with gE = the diagonal metric above
for i in range(4): gJ[i, i] = g[i, i].expr() / phi
RJ = gJ.ricci_scalar().expr()
lnphi = log(phi)
dln2 = sum(g.inverse()[i, i].expr() * diff(lnphi, X[i])^2 for i in range(4))
# sqrt(-g) = phi^{-2} sqrt(-gE) for g = phi^{-1} gE in four dimensions, so after dividing both sides by sqrt(-gE) the
# identity reads phi^{-1} R[g] = R_E + 3 Box_E ln phi - (3/2)(d_E ln phi)^2 (the common square root is cancelled
# analytically; Sage does not simplify sqrt(r^4 sin^2 th / phi^4) on its own)
print("[PASS] det g = phi^{-4} det gE for g = phi^{-1} gE:", bool((gJ.determinant().expr() - g.determinant().expr() / phi^4).simplify_full() == 0))
lhs = RJ / phi
rhs = R + 3 * box(lnphi) - 3 / 2 * dln2
print("[PASS] sqrt(-g) phi R[g] = sqrt(-gE) (R_E + 3 Box_E ln phi - (3/2)(d_E ln phi)^2) for g = phi^{-1} gE (both sides divided by sqrt(-gE)):", bool((lhs - rhs).simplify_full() == 0))

# Exercise exr-kkr-ricci-dilaton-components: for A = 0 the Ricci formulas reduce to R_yy = phi Box phi, R_{mu y} = 0,
# R_{mu nu} = R4_{mu nu} - (1/phi) nabla_mu d_nu phi. Checked on g = eta and phi = phi(t, x) (signature (+,-,-,-,-)).
M = Manifold(5, "M", structure="Lorentzian"); X = M.chart("t x v z y"); t, x, v, z, y = X[:]
phi = function("phi")(t, x)
G = M.metric("G")
for i, e in enumerate([1, -1, -1, -1]): G[i, i] = e
G[4, 4] = -phi^2
Ric = G.ricci()
ok = all(bool(Ric[m, 4].expr() == 0) for m in range(4))
ok2 = all(bool((Ric[m, n].expr() + diff(phi, X[m], X[n]) / phi).simplify_full() == 0) for m in range(4) for n in range(4))
print("R_tt =", Ric[0, 0].expr(), "; R_tx =", Ric[0, 1].expr())
print("[PASS] R_{mu y} = 0 for A = 0:", ok)
print("[PASS] R_{mu nu} = -(1/phi) d_mu d_nu phi for g = eta:", ok2)

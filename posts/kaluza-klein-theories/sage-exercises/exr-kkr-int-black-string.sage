# Exercise exr-kkr-int-black-string: the boosted black string and the Kaluza-Klein black hole. Start from the
# five-dimensional vacuum Schwarzschild x S^1, f dt^2 - dr^2/f - r^2 dOmega^2 - dw^2 with f = 1 - 2m/r, boost along w
# (t = c t' + s w', w = s t' + c w', c = cosh b, s = sinh b) and read off the Kaluza-Klein fields:
# phi^2 = 1 + (2m/r) s^2, A_t = (2m/r) s c/phi^2, Jordan metric g = (f/phi^2) dt^2 - dr^2/f - r^2 dOmega^2.
# Then check the five-dimensional vacuum equations as four-dimensional ones: R_yy = 0 is Box phi = -(1/4) phi^3 F^2,
# R_{mu y} - A_mu R_yy = 0 is nabla^l F_{l mu} = 3 phi^{-1} d_l phi F_mu^l, and R5 = 0 is R4 = -(3/4) phi^2 F^2.
m, b = var("m b"); assume(m > 0)
c, s = cosh(b), sinh(b)
M5 = Manifold(5, "M5", structure="Lorentzian"); X5 = M5.chart(r"t r th ph:\phi w"); t, r, th, ph, w = X5[:]
f = 1 - 2 * m / r
G = M5.metric("G")
G[0, 0] = f * c^2 - s^2; G[0, 4] = (f - 1) * s * c; G[4, 4] = f * s^2 - c^2
G[1, 1] = -1 / f; G[2, 2] = -r^2; G[3, 3] = -r^2 * sin(th)^2
print("[PASS] the boosted black string is Ricci flat:", all(bool(G.ricci()[i, j].expr().simplify_full() == 0) for i in range(5) for j in range(i, 5)))
phi2 = (-G[4, 4].expr()).simplify_full(); At = (-G[0, 4].expr() / phi2).simplify_full()
print("phi^2 =", phi2, "; A_t =", At)
print("[PASS] phi^2 = 1 + (2m/r) sinh^2 b and g_tt = G_tt + phi^2 A_t^2 = f/phi^2:", bool((phi2 - 1 - 2 * m / r * s^2).simplify_full() == 0) and bool((G[0, 0].expr() + phi2 * At^2 - f / phi2).simplify_full() == 0))
M = Manifold(4, "M", structure="Lorentzian"); X = M.chart(r"t r th ph:\phi"); t, r, th, ph = X[:]
P2 = (1 + 2 * m / r * s^2); phi = sqrt(P2); A_t = 2 * m / r * s * c / P2
g = M.metric("g"); g[0, 0] = f / P2; g[1, 1] = -1 / f; g[2, 2] = -r^2; g[3, 3] = -r^2 * sin(th)^2
gi = g.inverse(); nab = g.connection()
Af = M.one_form(); Af[0] = A_t
F = Af.exterior_derivative()
F2 = sum(F[i, j].expr() * F[k, l].expr() * gi[i, k].expr() * gi[j, l].expr() for i in range(4) for j in range(4) for k in range(4) for l in range(4))
boxphi = nab(nab(M.scalar_field(phi)).up(g)).trace().expr()
print("F^2 =", F2.simplify_full())
# phi Box phi = (1/2) Box phi^2 - (d phi)^2 with (d phi)^2 = (d phi^2)^2/(4 phi^2): rational in r, no square roots
boxP2 = nab(nab(M.scalar_field(P2)).up(g)).trace().expr()
dphi2 = sum(gi[i, j].expr() * diff(P2, X[i]) * diff(P2, X[j]) for i in range(4) for j in range(4)) / (4 * P2)
print("[PASS] R_yy = 0: phi Box phi = (1/2) Box phi^2 - (d phi)^2 = -(1/4) phi^4 F^2:", bool((boxP2 / 2 - dphi2 + P2^2 * F2 / 4).simplify_full() == 0))
Fu = F.up(g, 0).up(g, 1)                                 # F^{mu nu}
divF = nab(Fu).trace(0, 2)                                # nabla_l F^{l nu}
lhs = [sum(g[n, k].expr() * divF[k].expr() for k in range(4)) for n in range(4)]      # nabla^l F_{l mu}
Fmix = [[sum(F[mu, k].expr() * gi[k, l].expr() for k in range(4)) for l in range(4)] for mu in range(4)]
rhs = [3 / phi * sum(diff(phi, X[l]) * Fmix[mu][l] for l in range(4)) for mu in range(4)]
print("[PASS] R_{mu y} = A_mu R_yy: nabla^l F_{l mu} = 3 phi^{-1} d_l phi F_mu^l:", all(bool((lhs[mu] - rhs[mu]).simplify_full() == 0) for mu in range(4)))
R4 = g.ricci_scalar().expr()
print("[PASS] R5 = 0: R4 = -(3/4) phi^2 F^2:", bool((R4 + 3 / 4 * P2 * F2).simplify_full() == 0))

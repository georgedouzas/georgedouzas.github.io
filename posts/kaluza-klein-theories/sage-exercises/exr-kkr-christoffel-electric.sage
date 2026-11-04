# Exercise exr-kkr-christoffel-electric: the Christoffel symbols of the ansatz for a constant electric field. With
# g = eta (signature (+,-,-,-)), phi = 1 and A_x = B t (F_{tx} = B), SageManifolds' symbols are compared with the
# patterns of the text: Gamma^l_{mu y} = -(1/2) F_mu^l, Gamma^l_{yy} = 0, Gamma^l_{mu nu} = Gamma4 - (1/2)(A_mu F_nu^l
# + A_nu F_mu^l), and Gamma^y_{mu y} = -A_l Gamma^l_{mu y}.
M = Manifold(5, "M", structure="Lorentzian"); X = M.chart("t x v z y"); t, x, v, z, y = X[:]
B = var("B"); eta = [1, -1, -1, -1]
A = [0, B * t, 0, 0]
G = M.metric("G")
for i in range(4):
    for j in range(i, 4): G[i, j] = (eta[i] if i == j else 0) - A[i] * A[j]
    G[i, 4] = -A[i]
G[4, 4] = -1
Gam = G.christoffel_symbols(X)
F = matrix(SR, 4, 4, lambda m, n: diff(A[n], X[m]) - diff(A[m], X[n]))      # F_{mu nu}
Fmix = matrix(SR, 4, 4, lambda m, l: F[m, l] * eta[l])                        # F_mu^l = F_{mu k} eta^{k l}
ok1 = all(bool((Gam[l, m, 4].expr() - (-Fmix[m, l] / 2)).simplify_full() == 0) for l in range(4) for m in range(4))
ok2 = all(bool(Gam[l, 4, 4].expr() == 0) for l in range(4))
ok3 = all(bool((Gam[l, m, n].expr() - (-(A[m] * Fmix[n, l] + A[n] * Fmix[m, l]) / 2)).simplify_full() == 0) for l in range(4) for m in range(4) for n in range(4))
ok4 = all(bool((Gam[4, m, 4].expr() + sum(A[l] * Gam[l, m, 4].expr() for l in range(4))).simplify_full() == 0) for m in range(4))
print("Gamma^t_{x y} =", Gam[0, 1, 4].expr(), "; Gamma^x_{t y} =", Gam[1, 0, 4].expr())
print("[PASS] Gamma^l_{mu y} = -(1/2) F_mu^l:", ok1)
print("[PASS] Gamma^l_{yy} = phi d^l phi = 0:", ok2)
print("[PASS] Gamma^l_{mu nu} = -(1/2)(A_mu F_nu^l + A_nu F_mu^l) (Gamma4 = 0, phi = 1):", ok3)
print("[PASS] Gamma^y_{mu y} = d_mu phi/phi - A_l Gamma^l_{mu y} with phi = 1:", ok4)

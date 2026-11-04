# Exercise exr-kkr-ricci-electric: all components of the five-dimensional Ricci tensor for the constant electric
# field A_x = B t, phi = 1, g = eta, compared with the formula of the text: R_yy = (1/4) F^2, R_{mu y} = A_mu R_yy
# (nabla^l F_{l mu} = 0, d phi = 0), R_{mu nu} = (1/2) F_mu^l F_{nu l} + A_mu R_{nu y} + A_nu R_{mu y} - A_mu A_nu R_yy.
M = Manifold(5, "M", structure="Lorentzian"); X = M.chart("t x v z y"); t, x, v, z, y = X[:]
B = var("B"); eta = [1, -1, -1, -1]; A = [0, B * t, 0, 0]
G = M.metric("G")
for i in range(4):
    for j in range(i, 4): G[i, j] = (eta[i] if i == j else 0) - A[i] * A[j]
    G[i, 4] = -A[i]
G[4, 4] = -1
Ric = G.ricci()
F = matrix(SR, 4, 4, lambda m, n: diff(A[n], X[m]) - diff(A[m], X[n]))
F2 = sum(F[m, n]^2 * eta[m] * eta[n] for m in range(4) for n in range(4))
Ryy = F2 / 4
Rmy = [A[m] * Ryy for m in range(4)]
FF = matrix(SR, 4, 4, lambda m, n: sum(F[m, k] * eta[k] * F[n, k] for k in range(4)))   # F_mu^k F_{nu k}
ok = bool((Ric[4, 4].expr() - Ryy).simplify_full() == 0)
ok = ok and all(bool((Ric[m, 4].expr() - Rmy[m]).simplify_full() == 0) for m in range(4))
ok = ok and all(bool((Ric[m, n].expr() - (FF[m, n] / 2 + A[m] * Rmy[n] + A[n] * Rmy[m] - A[m] * A[n] * Ryy)).simplify_full() == 0) for m in range(4) for n in range(4))
print("R_tt =", Ric[0, 0].expr().simplify_full(), "; R_xx =", Ric[1, 1].expr().simplify_full(), "; R_xy =", Ric[1, 4].expr().simplify_full())
print("[PASS] the three Ricci formulas hold for the constant electric field:", ok)

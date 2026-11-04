# Exercise exr-kkr-scalar-plane-wave: an electromagnetic plane wave A_x = f(t - z) with phi = 1 and g = eta has
# F^2 = 0, so R5 = 0 and R_yy = 0 (no conflict with phi = 1), but R_{mu nu} is not zero: its frame component
# R_tt = (1/2) F_t^l F_{t l} = -(1/2) f'^2 is the energy density of the wave (with the sign of the signature).
M = Manifold(5, "M", structure="Lorentzian"); X = M.chart("t x v z y"); t, x, v, z, y = X[:]
f = function("f"); eta = [1, -1, -1, -1]; A = [0, f(t - z), 0, 0]
G = M.metric("G")
for i in range(4):
    for j in range(i, 4): G[i, j] = (eta[i] if i == j else 0) - A[i] * A[j]
    G[i, 4] = -A[i]
G[4, 4] = -1
Ric = G.ricci(); R5 = G.ricci_scalar().expr().simplify_full()
fp = diff(f(t - z), t)
print("R5 =", R5, "; R_yy =", Ric[4, 4].expr().simplify_full(), "; R_tt =", Ric[0, 0].expr().simplify_full())
print("[PASS] R5 = 0 and R_yy = 0 for the plane wave (F^2 = 0):", bool(R5 == 0) and bool(Ric[4, 4].expr().simplify_full() == 0))
print("[PASS] R_tt = (1/2) F_t^l F_{t l} = -(1/2) f'^2, not zero:", bool((Ric[0, 0].expr() + fp^2 / 2).simplify_full() == 0))

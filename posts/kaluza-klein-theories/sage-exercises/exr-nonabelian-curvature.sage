# Exercise exr-nonabelian-curvature: the six-dimensional curvature scalar of the ansatz for a single non-zero gauge
# field component A^1_x = c t (field strength F^1_{tx} = c) with SageManifolds, in the signature (+,-,-,-,-,-) with the
# internal block h = -gamma (gamma the round metric, R_h = -2): R_6 = -2 - (c^2/2) K_1 . K_1, which depends on the
# point of the sphere, and its average over the sphere, -2 - c^2/3.
th, ph = var("th ph")
K1 = vector([-sin(ph), -cot(th) * cos(ph)]); h = diagonal_matrix(SR, [1, sin(th)^2])
M = Manifold(6, "M", structure="Lorentzian"); X = M.chart("t x y z th6 ph6"); t, x, y, z, th6, ph6 = X[:]
c = var("c")
K1s = K1.subs(th=th6, ph=ph6); hs = h.subs(th=th6)
B = [[0, K1s[m] * c * t, 0, 0] for m in range(2)]
eta = [1, -1, -1, -1]
g = M.metric("g")                                            # internal block h = -gamma
for mu in range(4):
    for nu in range(mu, 4): g[mu, nu] = (eta[mu] if mu == nu else 0) - sum(hs[m, n] * B[m][mu] * B[n][nu] for m in range(2) for n in range(2))
    for m in range(2): g[mu, 4 + m] = -sum(hs[m, n] * B[n][mu] for n in range(2))
for m in range(2):
    for n in range(m, 2): g[4 + m, 4 + n] = -hs[m, n]
R6 = g.ricci_scalar().expr().simplify_full()
print("R_6 =", R6)
KK11 = (K1 * h * K1).simplify_full().subs(th=th6, ph=ph6)
print("[PASS] R_6 = -2 - (c^2/2) K_1 . K_1 (flat spacetime, R_h = -R_{S^2} = -2, F^2 = -2 c^2):", bool((R6 - (-2 - c^2 / 2 * KK11)).simplify_full() == 0))
avg = integrate(integrate(R6 * sin(th6), th6, 0, pi), ph6, 0, 2 * pi) / (4 * pi)
print("average of R_6 over the sphere:", avg.simplify_full())
print("[PASS] the average is -2 - c^2/3, i.e. -2 + (1/4)(2/3) F^2 with F^2 = -2 c^2:", bool((avg - (-2 - c^2 / 3)).simplify_full() == 0))

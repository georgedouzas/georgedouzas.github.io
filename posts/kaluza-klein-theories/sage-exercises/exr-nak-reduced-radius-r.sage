# Exercise exr-nak-reduced-radius-r (integrative): the reduced curvature scalar on a sphere of radius r,
# gamma = r^2 (dth^2 + sin^2 th dph^2), for A^1_x = c t, with SageManifolds in the signature (+,-,-,-,-,-):
# R_6 = -2/r^2 - (c^2/2) r^2 (1 - sin^2 th cos^2 ph), i.e. R - R_gamma + (1/4) K_1.K_1 F^1 F^1 with R_gamma = 2/r^2 and
# K_1 . K_1 = r^2 (1 - n_1^2); averaged over the sphere, -2/r^2 - c^2 r^2 / 3.
M = Manifold(6, "M", structure="Lorentzian"); X = M.chart("t x y z th ph"); t, x, y, z, th, ph = X[:]
c, r = var("c r"); assume(r > 0)
K1 = vector([-sin(ph), -cot(th) * cos(ph)])
gam = r^2 * diagonal_matrix(SR, [1, sin(th)^2])
B = [[0, K1[m] * c * t, 0, 0] for m in range(2)]
eta = [1, -1, -1, -1]
g = M.metric("g")
for mu in range(4):
    for nu in range(mu, 4): g[mu, nu] = (eta[mu] if mu == nu else 0) - sum(gam[m, n] * B[m][mu] * B[n][nu] for m in range(2) for n in range(2))
    for m in range(2): g[mu, 4 + m] = -sum(gam[m, n] * B[n][mu] for n in range(2))
for m in range(2):
    for n in range(m, 2): g[4 + m, 4 + n] = -gam[m, n]
R6 = g.ricci_scalar().expr().simplify_full()
print("R_6 =", R6)
print("[PASS] R_6 = -2/r^2 - (c^2/2) r^2 (1 - sin^2 th cos^2 ph):", bool((R6 - (-2 / r^2 - c^2 / 2 * r^2 * (1 - sin(th)^2 * cos(ph)^2))).simplify_full() == 0))
avg = integrate(integrate(R6 * sin(th), th, 0, pi), ph, 0, 2 * pi) / (4 * pi)
print("[PASS] the average over the sphere is -2/r^2 - c^2 r^2/3:", bool((avg - (-2 / r^2 - c^2 * r^2 / 3)).simplify_full() == 0))

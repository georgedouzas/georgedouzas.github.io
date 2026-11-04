# Exercise exr-nak-rhat-a3: the six-dimensional curvature scalar for the single gauge field A^3_x = c t along the
# Killing vector K_3 = d_ph of the unit sphere, with SageManifolds in the signature (+,-,-,-,-,-) (internal block
# h = -gamma): R_6 = -2 - (c^2/2) K_3 . K_3 = -2 - (c^2/2) sin^2 th, constant along the circles of latitude and
# largest in magnitude on the equator, where the orbits of K_3 are longest.
M = Manifold(6, "M", structure="Lorentzian"); X = M.chart("t x y z th ph"); t, x, y, z, th, ph = X[:]
c = var("c")
gam = diagonal_matrix(SR, [1, sin(th)^2])
B = [[0, 0, 0, 0], [0, c * t, 0, 0]]                  # B^m_mu = K^m_3 A^3_mu with K_3 = d_ph
eta = [1, -1, -1, -1]
g = M.metric("g")
for mu in range(4):
    for nu in range(mu, 4): g[mu, nu] = (eta[mu] if mu == nu else 0) - sum(gam[m, n] * B[m][mu] * B[n][nu] for m in range(2) for n in range(2))
    for m in range(2): g[mu, 4 + m] = -sum(gam[m, n] * B[n][mu] for n in range(2))
for m in range(2):
    for n in range(m, 2): g[4 + m, 4 + n] = -gam[m, n]
R6 = g.ricci_scalar().expr().simplify_full()
print("R_6 =", R6)
print("[PASS] R_6 = -2 - (c^2/2) sin^2 th:", bool((R6 - (-2 - c^2 / 2 * sin(th)^2)).simplify_full() == 0))
avg = integrate(integrate(R6 * sin(th), th, 0, pi), ph, 0, 2 * pi) / (4 * pi)
print("[PASS] its average over the sphere is -2 - c^2/3, the same as for A^1_x = c t:", bool((avg + 2 + c^2 / 3).simplify_full() == 0))

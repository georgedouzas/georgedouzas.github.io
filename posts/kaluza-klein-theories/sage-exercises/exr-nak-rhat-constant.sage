# Exercise exr-nak-rhat-constant: a constant non-abelian gauge field A^1_x = c1, A^2_y = c2 has no derivatives, but its
# field strength F^3_{xy} = eps_123 A^1_x A^2_y = c1 c2 is not zero. SageManifolds (signature (+,-,-,-,-,-), internal
# block h = -gamma) gives R_6 = -2 + (1/2) c1^2 c2^2 K_3 . K_3 = -2 + (1/2) c1^2 c2^2 sin^2 th: the geometry sees the
# commutator term, and F^a F^a = 2 (c1 c2)^2 > 0 for this magnetic field.
M = Manifold(6, "M", structure="Lorentzian"); X = M.chart("t x y z th ph"); t, x, y, z, th, ph = X[:]
c1, c2 = var("c1 c2")
K = [vector([-sin(ph), -cot(th) * cos(ph)]), vector([cos(ph), -cot(th) * sin(ph)]), vector([0, 1])]
gam = diagonal_matrix(SR, [1, sin(th)^2])
A = [[0, c1, 0, 0], [0, 0, c2, 0], [0, 0, 0, 0]]       # A^a_mu, mu = t, x, y, z
B = [[sum(K[a][m] * A[a][mu] for a in range(3)) for mu in range(4)] for m in range(2)]
eta = [1, -1, -1, -1]
g = M.metric("g")
for mu in range(4):
    for nu in range(mu, 4): g[mu, nu] = (eta[mu] if mu == nu else 0) - sum(gam[m, n] * B[m][mu] * B[n][nu] for m in range(2) for n in range(2))
    for m in range(2): g[mu, 4 + m] = -sum(gam[m, n] * B[n][mu] for n in range(2))
for m in range(2):
    for n in range(m, 2): g[4 + m, 4 + n] = -gam[m, n]
R6 = g.ricci_scalar().expr().simplify_full()
print("R_6 =", R6)
print("[PASS] R_6 = -2 + (1/2) c1^2 c2^2 sin^2 th:", bool((R6 - (-2 + c1^2 * c2^2 / 2 * sin(th)^2)).simplify_full() == 0))

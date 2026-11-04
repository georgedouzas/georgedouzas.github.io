# Exercise exr-gth-electroweak-masses: the doublet phi0 = (0, v) with D = d + g A^a T_a + g' B Y, T_a = -i sigma_a/2 and
# hypercharge Y = -i/2 (anti-Hermitian, hypercharge 1/2). The 4 x 4 mass matrix 2 Re[(t_A phi0)^dagger (t_B phi0)],
# t = (g T_1, g T_2, g T_3, g' Y), has eigenvalues g^2 v^2/2 (twice: W), (g^2 + g'^2) v^2/2 (Z) and 0 (the photon).
g, gp, v = var("g gp v", domain="positive")
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
t = [g * (-I / 2) * sa for sa in s] + [gp * (-I / 2) * identity_matrix(SR, 2)]
phi0 = vector(SR, [0, v])
M2 = matrix(SR, 4, 4, lambda A, B: 2 * ((t[A] * phi0).conjugate() * (t[B] * phi0)).real()).simplify_full()
print("mass matrix:"); print(M2)
ev = M2.eigenvalues()
print("eigenvalues:", ev)
print("[PASS] eigenvalues g^2 v^2/2 (twice), (g^2 + g'^2) v^2/2 and 0:", sorted(ev, key=str) == sorted([g^2 * v^2 / 2, g^2 * v^2 / 2, (g^2 + gp^2) * v^2 / 2, SR(0)], key=str))
ker = M2.right_kernel().basis()
print("massless combination (A^3, B) direction:", ker)
print("[PASS] the photon is proportional to g' A^3 + g B, i.e. sin(theta_W) A^3 + cos(theta_W) B with tan(theta_W) = g'/g:", len(ker) == 1 and bool(ker[0][0] == 0 and ker[0][1] == 0 and (ker[0][2] * g - ker[0][3] * gp).simplify_full() == 0))
print("[PASS] in Hermitian form the charge Q = sigma_3/2 + 1/2 (T_3 plus hypercharge 1/2) annihilates the vacuum:", ((s[2] / 2 + identity_matrix(SR, 2) / 2) * phi0).is_zero())
print("m_Z / m_W = sqrt(g^2 + g'^2)/g = 1/cos(theta_W)")

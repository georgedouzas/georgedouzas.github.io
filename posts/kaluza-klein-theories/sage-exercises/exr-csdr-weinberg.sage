# Exercise exr-csdr-weinberg: the Weinberg angle of the SU(3) over S^2 model. The gauge group H = SU(2) x U(1) sits in
# SU(3) with a single coupling g; the U(1) generator normalised like T_3, Y_n = lambda_8/2 with Tr Y_n^2 = 1/2, has
# eigenvalue sqrt(3)/2 on the doublet. In Standard Model conventions the Higgs doublet has Y = 1/2 with coupling g',
# so g' Y = g sqrt(3)/2 gives g' = sqrt(3) g and sin^2 theta_W = g'^2/(g^2 + g'^2) = 3/4 (Manton 1979).
lam8 = diagonal_matrix(SR, [1, 1, -2]) / sqrt(3); Yn = lam8 / 2
T3 = diagonal_matrix(SR, [1, -1, 0]) / 2
E13 = matrix(SR, 3, 3, lambda a, b: 1 if (a, b) == (0, 2) else 0)
ad = lambda A, B: A * B - B * A
print("Tr Y_n^2 =", (Yn * Yn).trace(), ", Tr T_3^2 =", (T3 * T3).trace(), ", Tr(Y_n T_3) =", (Yn * T3).trace())
print("[PASS] Y_n and T_3 are orthonormal in the SU(3) trace, Tr = 1/2 each, Tr(Y_n T_3) = 0:", bool((Yn * Yn).trace() == 1/2) and bool((T3 * T3).trace() == 1/2) and bool((Yn * T3).trace() == 0))
y = ad(Yn, E13)[0, 2]
print("hypercharge eigenvalue of the doublet under Y_n:", y)
g = var("g"); gp = g * y / (1/2)
print("[PASS] with Y = 1/2 for the doublet, g' = sqrt(3) g and sin^2 theta_W = 3/4:", bool(gp == sqrt(3) * g) and bool((gp^2 / (g^2 + gp^2)).simplify_full() == 3/4))
# the electric charge Q = T_3 + Y of the doublet components (0 and 1... in the convention Q = T_3 + Y with Y = 1/2): (1, 0)
Q = T3 + diagonal_matrix(SR, [1, 1, -2]) / 6                   # Y_SM = Y_n / sqrt(3)... as a 3x3 matrix: Y_SM acts on the doublet with 1/2
print("charges Q = T_3 + Y of the two doublet components:", [ad(Q, matrix(SR, 3, 3, lambda a, b: 1 if (a, b) == (i, 2) else 0))[i, 2] for i in range(2)])
print("[PASS] the doublet has electric charges (1, 0): the neutral component takes the vacuum value:", [ad(Q, matrix(SR, 3, 3, lambda a, b: 1 if (a, b) == (i, 2) else 0))[i, 2] for i in range(2)] == [1, 0])

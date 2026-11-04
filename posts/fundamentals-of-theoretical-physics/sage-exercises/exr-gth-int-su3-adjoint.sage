# Exercise exr-gth-int-su3-adjoint: an SU(3) adjoint Higgs Phi = v T_8 with L = -Tr(D Phi D Phi), D Phi = d Phi + g [A, Phi].
# The gauge-boson mass matrix M^2_AB = -2 g^2 Tr([T_A, Phi][T_B, Phi]) has four zero eigenvalues (SU(2) x U(1) unbroken)
# and four equal eigenvalues 3 g^2 v^2/4 (the gauge bosons of the broken generators T_4..T_7, which eat the four
# Goldstone bosons of the orbit SU(3)/(SU(2) x U(1)), of real dimension 4).
g, v = var("g v", domain="positive")
gm = [matrix(SR, 3, 3, d) for d in [
    [0, 1, 0, 1, 0, 0, 0, 0, 0], [0, -I, 0, I, 0, 0, 0, 0, 0], [1, 0, 0, 0, -1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 0, 0], [0, 0, -I, 0, 0, 0, I, 0, 0], [0, 0, 0, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, -I, 0, I, 0]]] + [matrix(SR, 3, 3, [1, 0, 0, 0, 1, 0, 0, 0, -2]) / sqrt(3)]
T = [-I / 2 * mm for mm in gm]
Phi = v * T[7]
com = [T[A] * Phi - Phi * T[A] for A in range(8)]
M2 = matrix(SR, 8, 8, lambda A, B: (-2 * g^2 * (com[A] * com[B]).trace()).simplify_full())
print("mass matrix (diagonal):", M2.diagonal()); print("[PASS] the mass matrix is diagonal:", M2 == diagonal_matrix(M2.diagonal()))
print("[PASS] four massless (T_1, T_2, T_3, T_8) and four of mass^2 3 g^2 v^2/4 (T_4..T_7):", [bool(d == 0) for d in M2.diagonal()] == [True] * 3 + [False] * 4 + [True] and all(bool(M2[A, A] == 3 * g^2 * v^2 / 4) for A in range(3, 7)))
print("[PASS] the kinetic normalisation: -Tr(D Phi D Phi) = (1/2) (d Phi^a)^2 since Tr(T_A T_B) = -delta/2:", all(bool((T[A] * T[B]).trace() == (-1 / 2 if A == B else 0)) for A in range(8) for B in range(8)))
print("broken generators:", 8 - 4, "= dim_R SU(3)/(SU(2) x U(1)) = dim_R CP^2 = 4")

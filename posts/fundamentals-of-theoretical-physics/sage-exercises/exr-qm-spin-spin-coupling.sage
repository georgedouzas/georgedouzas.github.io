# Exercise exr-qm-spin-spin-coupling: eigenvalues of (A/hbar^2) S1.S2 + (B/hbar)(S1z + S2z) on two spins 1/2.
hbar, A, B = var("hbar A B", domain="positive")
sig = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
one = identity_matrix(SR, 2)
S1 = [hbar / 2 * s.tensor_product(one) for s in sig]; S2 = [hbar / 2 * one.tensor_product(s) for s in sig]
H = A / hbar ** 2 * sum(a * b for a, b in zip(S1, S2)) + B / hbar * (S1[2] + S2[2])
print("eigenvalues:", H.eigenvalues())
J2 = sum((a + b) ** 2 for a, b in zip(S1, S2))
print("[H, J^2] = 0:", (H * J2 - J2 * H).simplify_full().is_zero(), "; [H, J_3] = 0:", (H * (S1[2] + S2[2]) - (S1[2] + S2[2]) * H).is_zero())

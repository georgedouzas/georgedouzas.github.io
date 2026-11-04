# Exercise exr-csd-doublet-transform: the action of the centraliser algebra h = s(u(2) + u(1)) on phi_+ = p E_13 + q E_23.
# For X = diag(U, u) with tr U + u = 0, [X, phi_+] = (U - u) (p, q)^T in the third column: the SU(2) generators act as
# on a doublet and tau acts with charge 1 (multiplication by i).
p, q = var("p q")
E = lambda i, j: matrix(SR, 3, 3, lambda a, b: 1 if (a, b) == (i, j) else 0)
ad = lambda A, B: A * B - B * A
Pp = p * E(0, 2) + q * E(1, 2)
sig = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
def up(m): M = zero_matrix(SR, 3, 3); M[0:2, 0:2] = m; return M
col = lambda M: vector(SR, [M[0, 2], M[1, 2]])
Phi = vector(SR, [p, q])
ok_su2 = all(bool((col(ad(I / 2 * up(s), Pp)) - (I / 2 * s) * Phi).is_zero()) for s in sig)
print("[PASS] [T_a, phi_+] corresponds to T_a Phi with T_a = (i/2) sigma_a: the doublet representation of SU(2):", ok_su2)
tau = I / 3 * diagonal_matrix(SR, [1, 1, -2])
print("[PASS] [tau, phi_+] corresponds to i Phi: charge 1 under the embedded U(1):", bool((col(ad(tau, Pp)) - I * Phi).is_zero()))
Qem = I / 3 * diagonal_matrix(SR, [2, -1, -1])
print("[Q_em, phi_+] corresponds to", col(ad(Qem, Pp)))
print("[PASS] [Q_em, phi_+] corresponds to (i p, 0): the upper component has electric charge 1, the lower 0:", bool((col(ad(Qem, Pp)) - vector(SR, [I * p, 0])).is_zero()))

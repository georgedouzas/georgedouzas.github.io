# Exercise exr-csd-fs-explicit: the internal field strength F_12 = [phi_1, phi_2] - tau on the doublet, as a matrix,
# its value at Phi = 0 and on the vacuum Phi = (0, 1), and Tr(tau F_12).
p, q, pb, qb = var("p q pb qb")
E = lambda i, j: matrix(SR, 3, 3, lambda a, b: 1 if (a, b) == (i, j) else 0)
ad = lambda A, B: A * B - B * A
tau = I / 3 * diagonal_matrix(SR, [1, 1, -2])
Pp = p * E(0, 2) + q * E(1, 2); Pm = -(pb * E(2, 0) + qb * E(2, 1))
f1 = (Pp + Pm) / 2; f2 = I * (Pp - Pm) / 2
F = (ad(f1, f2) - tau).expand()
print("F_12 ="); print(F)
r = p * pb + q * qb
print("[PASS] F_12 = (i/2) diag-block(Phi Phi^dagger - (2/3) 1, -|Phi|^2 + 4/3):",
      bool((F - I / 2 * matrix(SR, [[p * pb - 2/3, p * qb, 0], [q * pb, q * qb - 2/3, 0], [0, 0, -r + 4/3]])).expand().is_zero()))
print("[PASS] at Phi = 0, F_12 = -tau:", F.subs({p: 0, q: 0, pb: 0, qb: 0}) == -tau)
Qem = I / 3 * diagonal_matrix(SR, [2, -1, -1])
print("[PASS] at Phi = (0, 1), F_12 = -Q_em/2 = (i/6) diag(-2, 1, 1):", F.subs({p: 0, q: 1, pb: 0, qb: 1}) == -Qem / 2)
print("Tr(tau F_12) =", (tau * F).trace().expand())
print("[PASS] Tr(tau F_12) = 2/3 - r/2, and F_12 has rank-one upper block Phi Phi^dagger, so F_12 = 0 is impossible:", bool(((tau * F).trace() - (2/3 - r / 2)).expand() == 0))

# Exercise exr-csd-sol-bracket: the commutators of the doublet fields. [phi_+, phi_-] and [phi_1, phi_2] on the
# solution phi_+ = p E_13 + q E_23, in block form, and the check that both lie in the centraliser algebra.
p, q, pb, qb = var("p q pb qb")
E = lambda i, j: matrix(SR, 3, 3, lambda a, b: 1 if (a, b) == (i, j) else 0)
ad = lambda A, B: A * B - B * A
tau = I / 3 * diagonal_matrix(SR, [1, 1, -2])
Pp = p * E(0, 2) + q * E(1, 2); Pm = -(pb * E(2, 0) + qb * E(2, 1))
f1 = (Pp + Pm) / 2; f2 = I * (Pp - Pm) / 2
C = ad(Pp, Pm).expand(); B = ad(f1, f2).expand()
print("[phi_+, phi_-] ="); print(C); print("[phi_1, phi_2] ="); print(B)
print("[PASS] [phi_1, phi_2] = -(i/2) [phi_+, phi_-]:", bool((B + I / 2 * C).expand().is_zero()))
print("[PASS] both commutators commute with tau (they lie in h, as the grading predicts):", ad(tau, C).expand().is_zero() and ad(tau, B).expand().is_zero())
print("[PASS] [phi_1, phi_2] = (i/2) diag-block(Phi Phi^dagger, -|Phi|^2):",
      bool((B - I / 2 * matrix(SR, [[p * pb, p * qb, 0], [q * pb, q * qb, 0], [0, 0, -p * pb - q * qb]])).expand().is_zero()))

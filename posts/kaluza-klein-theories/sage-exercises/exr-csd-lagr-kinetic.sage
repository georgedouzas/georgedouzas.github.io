# Exercise exr-csd-lagr-kinetic: for two doublet configurations (phi_1, phi_2) built from Phi = (p, q) and
# Psi = (u, v), -sum_a Tr(phi_a psi_a) = Re(Phi^dagger Psi). Applied to D_mu phi_a, which is again of doublet form,
# this turns the scalar kinetic term -sum_a Tr(D phi_a D phi_a) into |D Phi|^2.
x = var("x1 y1 x2 y2 x3 y3 x4 y4", domain="real")
p, q, u, v = x[0] + I * x[1], x[2] + I * x[3], x[4] + I * x[5], x[6] + I * x[7]
E = lambda i, j: matrix(SR, 3, 3, lambda a, b: 1 if (a, b) == (i, j) else 0)
def fields(a, b):
    P = a * E(0, 2) + b * E(1, 2); M = -P.conjugate_transpose()
    return (P + M) / 2, I * (P - M) / 2
f1, f2 = fields(p, q); g1, g2 = fields(u, v)
lhs = (-(f1 * g1 + f2 * g2).trace()).simplify_full()
rhs = (p.conjugate() * u + q.conjugate() * v).real_part().simplify_full()
print("-sum_a Tr(phi_a psi_a) =", lhs)
print("[PASS] -sum_a Tr(phi_a psi_a) = Re(Phi^dagger Psi); in particular -sum_a Tr(phi_a^2) = |Phi|^2:", bool((lhs - rhs).simplify_full() == 0))
# D_mu phi_a = d phi_a + [A_mu, phi_a] with A_mu in h is of doublet form with Psi = d Phi + (U - u) Phi
a = var("a1 a2 a3 a4", domain="real")
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
def up(m): M = zero_matrix(SR, 3, 3); M[0:2, 0:2] = m; return M
A = sum(a[k] * I / 2 * up(s[k]) for k in range(3)) + a[3] * I / 3 * diagonal_matrix(SR, [1, 1, -2])
Avec = (sum(a[k] * I / 2 * s[k] for k in range(3)) + a[3] * I * identity_matrix(SR, 2)) * vector(SR, [p, q])
k1, k2 = fields(Avec[0], Avec[1])
c1 = A * f1 - f1 * A; c2 = A * f2 - f2 * A
print("[PASS] [A, phi_a] is the doublet field built from (T_a A^a + i A^4) Phi: the gauge field acts on Phi as on a doublet of hypercharge charge 1:",
      bool((c1 - k1).simplify_full().is_zero()) and bool((c2 - k2).simplify_full().is_zero()))

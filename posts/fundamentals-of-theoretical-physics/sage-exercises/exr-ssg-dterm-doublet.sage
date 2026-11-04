# Exercise exr-ssg-dterm-doublet: for an SU(2) doublet phi = (a1 + i b1, a2 + i b2) the D-terms D^a = -g phi^dag (sigma^a/2) phi
# give V_D = (1/2) sum_a (D^a)^2 = g^2 (phi^dag phi)^2 / 8, from the Fierz identity
# sum_a sigma^a_ij sigma^a_kl = 2 delta_il delta_jk - delta_ij delta_kl. For an adjoint field (three complex components
# phi^a) the D-term is D^a = -g (phi^dag T^a_adj phi) with (T^b)_ac = -i eps_bac, i.e. D = -i g phi^* x phi (cross product),
# which vanishes exactly when [phi, phi^dag] = 0: the D-flat directions of an adjoint are the commuting (normal) matrices.
g = var("g", domain="positive")
a1, b1, a2, b2 = var("a1 b1 a2 b2", domain="real")
phi = vector(SR, [a1 + I * b1, a2 + I * b2])
pauli = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
D = [-g * (phi.conjugate() * (p / 2) * phi) for p in pauli]
print("[PASS] each D^a is real:", all(bool(d.imag_part().simplify_full() == 0) for d in D))
VD = sum(d^2 for d in D) / 2
n2 = (phi.conjugate() * phi).expand()
print("[PASS] V_D = g^2 (phi^dag phi)^2 / 8:", bool((VD - g^2 * n2^2 / 8).expand().simplify_full() == 0))
fierz = all(sum(p[i, j] * p[k, l] for p in pauli) == 2 * (i == l) * (j == k) - (i == j) * (k == l) for i in range(2) for j in range(2) for k in range(2) for l in range(2))
print("[PASS] Fierz identity sum_a sigma^a_ij sigma^a_kl = 2 delta_il delta_jk - delta_ij delta_kl:", fierz)
# adjoint: phi = phi^a T^a with T^a = sigma^a/2 as a 2x2 matrix; D^a proportional to tr(T^a [phi, phi^dag])
u = [var("u%d" % k, domain="real") for k in range(3)]; v = [var("v%d" % k, domain="real") for k in range(3)]
pa = [u[k] + I * v[k] for k in range(3)]
Phi = sum(pa[k] * pauli[k] / 2 for k in range(3))
C = Phi * Phi.conjugate_transpose() - Phi.conjugate_transpose() * Phi
Dadj = [(-I * sum(pa[b].conjugate() * pa[c] * (1 if (a, b, c) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)] else -1 if (a, c, b) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)] else 0) for b in range(3) for c in range(3))) for a in range(3)]
print("[PASS] the adjoint D-terms phi^* T^a_adj phi = -i (phi^* x phi)^a equal 2 tr(T^a [phi, phi^dag]) up to sign:",
      all(bool((Dadj[a] - 2 * (pauli[a] / 2 * C).trace()).expand() == 0) or bool((Dadj[a] + 2 * (pauli[a] / 2 * C).trace()).expand() == 0) for a in range(3)))
print("[PASS] a real direction phi^a = u^a (v = 0) is D-flat: [phi, phi^dag] = 0:", C.subs({v[k]: 0 for k in range(3)}).expand().is_zero())

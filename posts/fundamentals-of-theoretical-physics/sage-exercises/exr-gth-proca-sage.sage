# Exercise exr-gth-proca-sage: the plane wave A^nu = eps^nu exp(-i k.x) solves the Proca equation
# d_mu F^{mu nu} + m^2 A^nu = 0 (signature (+, -, -, -)) iff M eps = 0 with M^nu_rho = (m^2 - k^2) delta^nu_rho + k^nu k_rho;
# det M = m^2 (m^2 - k^2)^3, and on the mass shell k^2 = m^2 the solutions are the three polarisations with k.eps = 0.
t, x, y, z = var("t x y z", domain="real"); X = [t, x, y, z]
m = var("m", domain="positive")
k = vector(SR, var("k0 k1 k2 k3", domain="real")); ep = vector(SR, var("e0 e1 e2 e3"))
eta = diagonal_matrix([1, -1, -1, -1])
kdn = eta * k; kx = sum(kdn[i] * X[i] for i in range(4))
Aup = ep * exp(-I * kx); Adn = eta * Aup
F = matrix(SR, 4, 4, lambda a, b: diff(Adn[b], X[a]) - diff(Adn[a], X[b])); Fup = eta * F * eta
proca = vector(SR, [sum(diff(Fup[a, n], X[a]) for a in range(4)) + m^2 * Aup[n] for n in range(4)])
k2 = k * eta * k
M = (m^2 - k2) * identity_matrix(SR, 4) + k.column() * kdn.row()
print("[PASS] the Proca operator on the plane wave is (M eps) exp(-i k.x):", ((proca - M * ep * exp(-I * kx)).apply_map(lambda e: e.simplify_full())).is_zero())
print("[PASS] det M = m^2 (m^2 - k^2)^3:", bool((M.det() - m^2 * (m^2 - k2)^3).expand() == 0))
Mrest = M.subs({k[0]: m, k[1]: 0, k[2]: 0, k[3]: 0})
print("rest frame k = (m, 0, 0, 0): M =", Mrest.list())
print("[PASS] on shell in the rest frame the solutions are eps = (0, e1, e2, e3): three polarisations:", Mrest.right_kernel().dimension() == 3 and all(w[0] == 0 for w in Mrest.right_kernel().basis()))

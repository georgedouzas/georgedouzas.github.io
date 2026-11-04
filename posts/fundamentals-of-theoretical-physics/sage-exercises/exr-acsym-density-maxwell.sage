# Exercise exr-acsym-density-maxwell: the Maxwell Lagrangian -(1/4) F_{mu nu} F^{mu nu} in terms of the electric and
# magnetic fields, signature (+, -, -, -). A^mu = (A^0, vec A), A_mu = eta_{mu nu} A^nu, F_{mu nu} = d_mu A_nu - d_nu A_mu,
# E = -grad A^0 - d_t vec A, B = curl vec A.
t, x, y, z = var('t x y z', domain='real'); X = [t, x, y, z]
eta = diagonal_matrix(SR, [1, -1, -1, -1])
Aup = [function('A%d' % mu)(*X) for mu in range(4)]                          # A^mu
Adn = [sum(eta[mu, nu] * Aup[nu] for nu in range(4)) for mu in range(4)]      # A_mu
F = matrix(SR, 4, 4, lambda m, n: diff(Adn[n], X[m]) - diff(Adn[m], X[n]))    # F_{mu nu}
Fup = eta * F * eta                                                           # F^{mu nu}
L = -sum(F[m, n] * Fup[m, n] for m in range(4) for n in range(4)) / 4
E = vector([-diff(Aup[0], X[i]) - diff(Aup[i], t) for i in (1, 2, 3)])
A = vector(Aup[1:])
B = vector([diff(A[2], y) - diff(A[1], z), diff(A[0], z) - diff(A[2], x), diff(A[1], x) - diff(A[0], y)])
print("F_{0i} = E^i:", all(bool((F[0, i] - E[i - 1]).expand() == 0) for i in (1, 2, 3)))
print("F_{12} = -B^3, F_{23} = -B^1, F_{31} = -B^2:", bool((F[1, 2] + B[2]).expand() == 0) and bool((F[2, 3] + B[0]).expand() == 0) and bool((F[3, 1] + B[1]).expand() == 0))
print("F_{mu nu} F^{mu nu} = 2 (B^2 - E^2):", bool((sum(F[m, n] * Fup[m, n] for m in range(4) for n in range(4)) - 2 * (B * B - E * E)).expand() == 0))
print("-(1/4) F_{mu nu} F^{mu nu} = (E^2 - B^2)/2:", bool((L - (E * E - B * B) / 2).expand() == 0))

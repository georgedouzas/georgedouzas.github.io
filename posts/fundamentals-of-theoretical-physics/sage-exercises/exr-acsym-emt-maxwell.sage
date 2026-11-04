# Exercise exr-acsym-emt-maxwell: the energy-momentum tensor T^{mu nu} = F^{mu rho} F_rho^nu + (1/4) eta^{mu nu} F^2 of
# the electromagnetic field in terms of E and B, signature (+, -, -, -), with F_{0i} = E_i and F_{ij} = -eps_{ijk} B_k.
E = vector(var('E1 E2 E3')); B = vector(var('B1 B2 B3'))
eta = diagonal_matrix(SR, [1, -1, -1, -1])
eps3 = lambda i, j, k: (i - j) * (j - k) * (k - i) / 2                   # Levi-Civita symbol for indices 0, 1, 2
F = matrix(SR, 4, 4)
for i in range(3):
    F[0, i + 1] = E[i]; F[i + 1, 0] = -E[i]
    for j in range(3):
        F[i + 1, j + 1] = -sum(eps3(i, j, k) * B[k] for k in range(3))
Fup = eta * F * eta                                                    # F^{mu nu}
F2 = sum(F[m, n] * Fup[m, n] for m in range(4) for n in range(4))
T = Fup * eta * Fup + eta * F2 / 4                                      # F^{mu rho} eta_{rho sigma} F^{sigma nu} + (1/4) eta^{mu nu} F^2
print("T^{00} = (E^2 + B^2)/2:", bool((T[0, 0] - (E * E + B * B) / 2).expand() == 0))
print("T^{0i} = (E x B)_i, the Poynting vector:", all(bool((T[0, i + 1] - E.cross_product(B)[i]).expand() == 0) for i in range(3)))
print("T^{ij} = -E_i E_j - B_i B_j + delta_ij (E^2 + B^2)/2, the Maxwell stress:", all(bool((T[i + 1, j + 1] - (-E[i] * E[j] - B[i] * B[j] + (1 if i == j else 0) * (E * E + B * B) / 2)).expand() == 0) for i in range(3) for j in range(3)))
print("symmetric:", T == T.transpose(), "; traceless, eta_{mu nu} T^{mu nu} =", sum(eta[m, n] * T[m, n] for m in range(4) for n in range(4)).expand())

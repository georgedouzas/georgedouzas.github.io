# Exercise exr-gth-maxwell-from-fields: with A^mu = (Phi, A_vec), A_mu = eta_{mu nu} A^nu = (Phi, -A_vec) in the
# signature (+, -, -, -), E = -grad Phi - d_t A_vec and B = curl A_vec, check -1/4 F_{mu nu} F^{mu nu} = (E^2 - B^2)/2.
t, x, y, z = var("t x y z", domain="real"); X = [t, x, y, z]
Phi, Ax, Ay, Az = [function(n)(t, x, y, z) for n in ["Phi", "Ax", "Ay", "Az"]]
eta = diagonal_matrix([1, -1, -1, -1])
Aup = vector(SR, [Phi, Ax, Ay, Az]); Adn = eta * Aup
F = matrix(SR, 4, 4, lambda m, n: diff(Adn[n], X[m]) - diff(Adn[m], X[n]))
Fup = eta * F * eta
lag = -sum(F[m, n] * Fup[m, n] for m in range(4) for n in range(4)) / 4
Av = vector(SR, [Ax, Ay, Az])
E = -vector(SR, [diff(Phi, w) for w in (x, y, z)]) - diff(Av, t)
B = vector(SR, [diff(Az, y) - diff(Ay, z), diff(Ax, z) - diff(Az, x), diff(Ay, x) - diff(Ax, y)])
print("[PASS] -1/4 F_{mu nu} F^{mu nu} = (E^2 - B^2)/2:", bool((lag - (E * E - B * B) / 2).expand() == 0))
print("[PASS] F^{i0} = E^i (the electric field sits in the time column):", all(bool((Fup[i + 1, 0] - E[i]).expand() == 0) for i in range(3)))
print("[PASS] F_{12} = -B^3 and its cyclic partners:", bool((F[1, 2] + B[2]).expand() == 0) and bool((F[2, 3] + B[0]).expand() == 0) and bool((F[3, 1] + B[1]).expand() == 0))

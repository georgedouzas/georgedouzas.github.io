# Exercise exr-csd-eq-structure: the structure constants of su(2) in the basis Q_a = sigma_a/(2i), the action of
# ad Q_3 on the tangent directions Q_1, Q_2 and on Q_+- = Q_1 -+ i Q_2, read as the constraint equations.
Q = [matrix(SR, [[0, 1], [1, 0]]) / (2 * I), matrix(SR, [[0, -I], [I, 0]]) / (2 * I), matrix(SR, [[1, 0], [0, -1]]) / (2 * I)]
ad = lambda A, B: A * B - B * A
eps = lambda a, b, c: sign(Permutation([a + 1, b + 1, c + 1])) if len({a, b, c}) == 3 else 0
print("[PASS] [Q_a, Q_b] = eps_abc Q_c:", all(ad(Q[a], Q[b]) == sum(eps(a, b, c) * Q[c] for c in range(3)) for a in range(3) for b in range(3)))
print("[PASS] [Q_3, Q_1] = Q_2 and [Q_3, Q_2] = -Q_1: f_31^2 = 1, f_32^1 = -1:", ad(Q[2], Q[0]) == Q[1] and ad(Q[2], Q[1]) == -Q[0])
Qp = Q[0] - I * Q[1]; Qm = Q[0] + I * Q[1]
print("Q_+ =", Qp.list(), ", Q_- =", Qm.list())
print("[PASS] [Q_3, Q_+] = i Q_+ and [Q_3, Q_-] = -i Q_-: the constraints say phi_+- = phi_1 -+ i phi_2 has tau-charge +-1:", ad(Q[2], Qp) == I * Qp and ad(Q[2], Qm) == -I * Qm)
print("[PASS] [Q_1, Q_2] = Q_3: f_12^3 = 1, the constant entering F_12 = [phi_1, phi_2] - phi_3:", ad(Q[0], Q[1]) == Q[2])

# Exercise exr-la-field-strength (integrative): an antisymmetric type (0, 2) tensor F_{mu nu} on Minkowski space with
# electric field E = (E1, 0, 0) and magnetic field B = (0, 0, B3): raise both indices with eta^{mu nu} = eta_{mu nu} and
# compute the invariant F_{mu nu} F^{mu nu} = 2 (B^2 - E^2).
E1, B3 = var("E1 B3")
eta = diagonal_matrix(SR, [1, -1, -1, -1])
F = matrix(SR, [[0, -E1, 0, 0], [E1, 0, B3, 0], [0, -B3, 0, 0], [0, 0, 0, 0]])
print("F antisymmetric:", F.transpose() == -F)
Fup = eta.inverse() * F * eta.inverse()           # F^{mu nu} = eta^{mu a} F_{ab} eta^{b nu}
print("F^{mu nu} =", Fup.rows())
inv = sum(F[m, n] * Fup[m, n] for m in range(4) for n in range(4)).expand()
print("F_{mu nu} F^{mu nu} =", inv, "= 2 (B3^2 - E1^2):", bool(inv == 2 * (B3 ^ 2 - E1 ^ 2)))

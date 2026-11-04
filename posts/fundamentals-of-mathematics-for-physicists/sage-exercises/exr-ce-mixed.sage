# Exercise exr-ce-mixed: the Hurwitz zeta value zeta_H(-1, 1/2), its Bernoulli-polynomial form, the identity
# sum (n + 1/2)^(-s) = (2^s - 1) zeta(s) at s = -1, and the repulsive Dirichlet-Neumann Casimir energy pi/(48 L).
zh = hurwitz_zeta(-1, QQ(1) / 2)
B2 = bernoulli_polynomial(QQ(1) / 2, 2)
print("zeta_H(-1, 1/2) =", zh, "; -B_2(1/2)/2 =", -B2 / 2, "; (2^-1 - 1) zeta(-1) =", (QQ(1) / 2 - 1) * zeta(-1))
L = var("L", domain="positive")
E = pi / (2 * L) * zh
print("E =", E, "; F = -dE/dL =", -diff(E, L))

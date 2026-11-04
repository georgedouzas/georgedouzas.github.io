# Exercise exr-kkf-gamma-contraction: the two contributions of the spin connection of the Kaluza-Klein frame to
# Gamma^C (1/4) omega_AB(E_C) Gamma^A Gamma^B, with Gamma^mu = gamma^mu (Dirac representation) and Gamma^4 = i gamma_5,
# for a random antisymmetric F_ab, and the chiral rotation of the Pauli term.
import random
random.seed(int(7))
s0 = matrix(QQbar, [[1, 0], [0, 1]]); s1 = matrix(QQbar, [[0, 1], [1, 0]]); s2 = matrix(QQbar, [[0, -I], [I, 0]]); s3 = matrix(QQbar, [[1, 0], [0, -1]])
Z = matrix(QQbar, 2, 2); blk = lambda a, b, c, d: block_matrix([[a, b], [c, d]], subdivide=False)
g = [blk(s0, Z, Z, -s0)] + [blk(Z, s, -s, Z) for s in (s1, s2, s3)]
g5 = I * g[0] * g[1] * g[2] * g[3]; G = g + [I * g5]; one = identity_matrix(QQbar, 4)
F = matrix(QQ, 4, 4)
for a in range(4):
    for b in range(a + 1, 4):
        F[a, b] = QQ(random.randint(int(-6), int(6))); F[b, a] = -F[a, b]
print("F_ab =", F.rows())
FG = sum(F[a, b] * G[a] * G[b] for a in range(4) for b in range(4))
part1 = G[4] * sum(F[a, b] / 2 * G[a] * G[b] for a in range(4) for b in range(4)) / 4          # omega_ab(E_4) = F_ab/2
part2 = sum(G[c] * sum(-F[a, c] / 2 * G[4] * G[a] + F[a, c] / 2 * G[a] * G[4] for a in range(4)) for c in range(4)) / 4   # omega_4a(E_c), omega_a4(E_c)
print("omega_ab(E_4) part = +(1/8) F g g Gamma^4:", part1 == FG * G[4] / 8)
print("omega_4a(E_b) part = -(1/4) F g g Gamma^4:", part2 == -FG * G[4] / 4)
print("total = -(1/8) F g g Gamma^4:", part1 + part2 == -FG * G[4] / 8)
sigF = sum(F[a, b] * I / 2 * (G[a] * G[b] - G[b] * G[a]) for a in range(4) for b in range(4))
rot = (one - I * g5) / sqrt(QQbar(2))
print("exp(-i pi g5/4) is (1 - i g5)/sqrt(2):", rot * rot == -I * g5)
print("rotated Pauli term = -(1/8) sigma^ab F_ab:", rot * (FG * g5 / 8) * rot == -sigF / 8)

# Exercise exr-vf-so3-brackets: the Lie brackets of the three rotation fields of the sphere with SageManifolds' bracket:
# [K_a, K_b] = -epsilon_abc K_c, and the Jacobi identity.
S2 = Manifold(2, "S^2"); A = S2.open_subset("A")
sph = A.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi"); th, ph = sph[:]
K1 = A.vector_field(-sin(ph), -cos(th)/sin(th)*cos(ph), name="K1")
K2 = A.vector_field(cos(ph), -cos(th)/sin(th)*sin(ph), name="K2")
K3 = A.vector_field(0, 1, name="K3")
K = [K1, K2, K3]
for (a, b, c) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
    print("[K%d, K%d] = -K%d:" % (a + 1, b + 1, c + 1), K[a].bracket(K[b]) == -K[c])
J = K1.bracket(K2.bracket(K3)) + K2.bracket(K3.bracket(K1)) + K3.bracket(K1.bracket(K2))
print("Jacobi identity [K1,[K2,K3]] + [K2,[K3,K1]] + [K3,[K1,K2]] = 0:", J == 0)

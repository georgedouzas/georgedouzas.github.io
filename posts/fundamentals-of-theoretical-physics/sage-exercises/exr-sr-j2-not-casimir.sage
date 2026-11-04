# Exercise exr-sr-j2-not-casimir: J^2 does not commute with the boosts in the four-vector representation, while J^2 - K^2
# and J.K do commute with all generators.
eps = lambda i, j, k: sign(Permutation([i + 1, j + 1, k + 1]).signature()) if len({i, j, k}) == 3 else 0
J = [matrix(QQ, 4, 4, lambda a, b: -eps(i, a - 1, b - 1) if a > 0 and b > 0 else 0) for i in range(3)]
K = [matrix(QQ, 4, 4, lambda a, b: 1 if (a, b) in [(0, i + 1), (i + 1, 0)] else 0) for i in range(3)]
J2 = sum(j^2 for j in J); C1 = J2 - sum(k^2 for k in K); C2 = sum(J[i]*K[i] for i in range(3))
print("[K1, J^2] ="); print(K[0]*J2 - J2*K[0])
gens = J + K
print("J^2 - K^2 commutes with all generators:", all(g*C1 == C1*g for g in gens), " value:", C1.diagonal())
print("J.K commutes with all generators:", all(g*C2 == C2*g for g in gens), " value:", C2.diagonal())

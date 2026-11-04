# Exercise exr-prob-conditional-table: conditional probabilities of "sum >= 10" and "sum = 7" given the first die,
# as exact rationals, and the law of total probability for both events.
Omega = [(i, j) for i in range(1, 7) for j in range(1, 7)]
P = lambda E: QQ(len(E)) / len(Omega)
A = [w for w in Omega if w[0] + w[1] >= 10]
C = [w for w in Omega if w[0] + w[1] == 7]
rows = {i: [w for w in Omega if w[0] == i] for i in range(1, 7)}
cond = lambda E, F: P([w for w in E if w in F]) / P(F)
for i in range(1, 7):
    print("first die =", i, ": P(sum >= 10 | i) =", cond(A, rows[i]), "  P(sum = 7 | i) =", cond(C, rows[i]))
totA = sum(cond(A, rows[i]) * P(rows[i]) for i in range(1, 7))
totC = sum(cond(C, rows[i]) * P(rows[i]) for i in range(1, 7))
print("total probability: sum_i P(A | i) P(i) =", totA, "= P(A) =", P(A), ":", totA == P(A))
print("total probability: sum_i P(C | i) P(i) =", totC, "= P(C) =", P(C), ":", totC == P(C))

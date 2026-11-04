# Exercise exr-prob-inclusion-exclusion-three: inclusion-exclusion for three events on two dice: first die even,
# second die even, sum at least 8; exact rational probabilities.
Omega = [(i, j) for i in range(1, 7) for j in range(1, 7)]
P = lambda E: QQ(len(E)) / len(Omega)
A = [w for w in Omega if w[0] % 2 == 0]
B = [w for w in Omega if w[1] % 2 == 0]
C = [w for w in Omega if w[0] + w[1] >= 8]
I = lambda *Es: [w for w in Omega if all(w in E for E in Es)]
union = [w for w in Omega if w in A or w in B or w in C]
rhs = P(A) + P(B) + P(C) - P(I(A, B)) - P(I(A, C)) - P(I(B, C)) + P(I(A, B, C))
print("P(A), P(B), P(C) =", P(A), P(B), P(C))
print("P(AB), P(AC), P(BC), P(ABC) =", P(I(A, B)), P(I(A, C)), P(I(B, C)), P(I(A, B, C)))
print("P(A or B or C) =", P(union), "; inclusion-exclusion gives", rhs, ":", P(union) == rhs)

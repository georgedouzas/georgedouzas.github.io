# Exercise exr-ems-capacitance-three-shells (Pierrus, Question 3.12): coefficients of potential and capacitance of three shells.
r = [1, 2, 3]
P = matrix(QQ, 3, 3, lambda i, j: 1 / max(r[i], r[j])) / (4 * pi)
C = P.inverse().simplify_full()
print("4 pi P =", (4 * pi * P).simplify_full())
print("C/(4 pi) =", (C / (4 * pi)).simplify_full())
print("symmetric:", C == C.transpose(), "; diagonal positive, off-diagonal <= 0:", all(C[i, i] > 0 for i in range(3)) and all(C[i, j] <= 0 for i in range(3) for j in range(3) if i != j))

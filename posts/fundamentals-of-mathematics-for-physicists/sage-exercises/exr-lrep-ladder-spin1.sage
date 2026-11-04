# Exercise exr-lrep-ladder-spin1: the ladder operators J_3 = i L_3, J_+- = i L_1 -+ L_2 of the vector representation,
# their brackets, and the chain of weights 1, 0, -1 they connect.
L = [matrix(QQbar, 3, 3, lambda b, c: -sign(Permutation([a + 1, b + 1, c + 1])) if len({a, b, c}) == 3 else 0) for a in range(3)]
J3 = I * L[2]; Jp = I * L[0] - L[1]; Jm = I * L[0] + L[1]
br = lambda A, B: A * B - B * A
print("[J3, J+] = J+:", br(J3, Jp) == Jp, " [J3, J-] = -J-:", br(J3, Jm) == -Jm, " [J+, J-] = 2 J3:", br(Jp, Jm) == 2 * J3)
v = {ZZ(ev): vecs[0] for ev, vecs, m in J3.eigenvectors_right()}
for m in [-1, 0, 1]:
    w = Jp * v[m]
    print(f"J+ on the weight-{m} vector:", "zero" if w == 0 else f"weight {m + 1}: {J3 * w == (m + 1) * w}")
print("J+ is nilpotent of order 3 (a chain of three weights):", Jp^2 != 0 and Jp^3 == 0)

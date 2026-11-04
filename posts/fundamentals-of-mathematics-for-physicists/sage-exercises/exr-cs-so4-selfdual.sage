# Exercise exr-cs-so4-selfdual: so(4) = su(2) + su(2). With J^{+-}_a = (1/2)((1/2) eps_abc M_bc +- M_a3) for
# a, b, c in {0, 1, 2}, the two triples commute with each other and each closes into a copy of so(3); the isotropy
# algebra r = span(M_01, M_02, M_12) is the diagonal and the coset k = span(M_03, M_13, M_23) the antidiagonal.
def M(i, j):
    m = matrix(QQ, 4, 4)
    if i == j: return m
    m[i, j] = 1; m[j, i] = -1; return m
eps = lambda a, b, c: sign(Permutation([a + 1, b + 1, c + 1])) if len({a, b, c}) == 3 else 0
J = {s: [(sum(eps(a, b, c) * M(b, c) for b in range(3) for c in range(3)) / 2 + s * M(a, 3)) / 2 for a in range(3)] for s in (1, -1)}
br = lambda X, Y: X * Y - Y * X
print("[J+_a, J-_b] = 0 for all a, b:", all(br(J[1][a], J[-1][b]) == 0 for a in range(3) for b in range(3)))
for s in (1, -1):
    for c0 in (1, -1):
        if all(br(J[s][a], J[s][b]) == c0 * sum(eps(a, b, c) * J[s][c] for c in range(3)) for a in range(3) for b in range(3)):
            print(f"[J{'+' if s == 1 else '-'}_a, J{'+' if s == 1 else '-'}_b] = {c0} * eps_abc J{'+' if s == 1 else '-'}_c")
print("J+_a + J-_a = (1/2) eps_abc M_bc (in r):", [J[1][a] + J[-1][a] for a in range(3)] == [M(1, 2), M(2, 0), M(0, 1)])
print("J+_a - J-_a = M_a3 (in k):", [J[1][a] - J[-1][a] for a in range(3)] == [M(0, 3), M(1, 3), M(2, 3)])

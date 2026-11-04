# Exercise exr-cs-s4-coset: S^4 = SO(5)/SO(4). Build the ten generators M_ij of so(5), split them into r (i, j < 4)
# and k (the M_i4), and sort all brackets: [r, r] in r, [r, k] in k, [k, k] in r, so S^4 is a symmetric coset of
# dimension 10 - 6 = 4 with [M_a4, M_b4] = -M_ab.
n = 5
def M(i, j):
    m = matrix(QQ, n, n); m[i, j] = 1; m[j, i] = -1; return m
r = [(i, j) for i in range(4) for j in range(i + 1, 4)]; k = [(i, 4) for i in range(4)]
inr = lambda X: all(X[i, 4] == 0 for i in range(4))          # no component along the M_i4
ink = lambda X: all(X[i, j] == 0 for (i, j) in r)
br = lambda X, Y: X * Y - Y * X
print("dim so(5) =", len(r) + len(k), " dim so(4) =", len(r), " dim S^4 =", len(k))
print("[r, r] in r:", all(inr(br(M(*p), M(*q))) for p in r for q in r))
print("[r, k] in k (reductive):", all(ink(br(M(*p), M(*q))) for p in r for q in k))
print("[k, k] in r (symmetric):", all(inr(br(M(*p), M(*q))) for p in k for q in k))
print("[M_a4, M_b4] = -M_ab for all a < b:", all(br(M(a, 4), M(b, 4)) == -M(a, b) for a in range(4) for b in range(a + 1, 4)))
print("number of vanishing brackets [r, k]:", sum(1 for p in r for q in k if br(M(*p), M(*q)) == 0), "of", len(r) * len(k))

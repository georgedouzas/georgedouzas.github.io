# Exercise exr-lrep-root-sun: the roots of su(4): for diagonal H = diag(h_1, ..., h_4), [H, E_ij] = (h_i - h_j) E_ij on
# the matrix units, so the roots are e_i - e_j (i != j); compare with Sage's RootSystem(["A", 3]).
n = 4
h = vector(SR, var("h1 h2 h3 h4"))
H = diagonal_matrix(SR, list(h))
roots = []
for i in range(n):
    for j in range(n):
        if i != j:
            E = matrix(SR, n, n, {(i, j): 1})
            assert bool(H * E - E * H == (h[i] - h[j]) * E)
            roots.append(tuple(1 if k == i else (-1 if k == j else 0) for k in range(n)))
print("number of roots from the matrix units:", len(roots), "= n(n - 1) =", n * (n - 1))
L = RootSystem(["A", 3]).ambient_space()
print("same set as Sage's roots of A3:", set(roots) == set(tuple(r.to_vector()) for r in L.roots()))
print("dim su(4) = rank + #roots =", (n - 1) + len(roots))

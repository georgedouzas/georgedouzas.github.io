# Exercise exr-repn-invariant-z3: the cyclic permutation C of the coordinates of R^3 (a representation of Z_3):
# its invariant subspaces over Q and its eigenlines over the cyclotomic field Q(w), w = exp(2 pi i/3).
C = matrix(QQ, [[0, 0, 1], [1, 0, 0], [0, 1, 0]])      # e1 -> e2 -> e3 -> e1
print("C^3 = 1:", C**3 == 1)
print("rational invariant subspaces: kernel of C - 1 =", (C - 1).right_kernel().basis(), "; kernel of C^2 + C + 1 has dimension", (C**2 + C + 1).right_kernel().dimension())
K.<w> = CyclotomicField(3)
CK = C.change_ring(K)
for ev, vecs, mult in CK.eigenvectors_right():
    print("  eigenvalue", ev, "with eigenvector", vecs[0])
print("characteristic polynomial:", C.charpoly(), "=", C.charpoly().factor())

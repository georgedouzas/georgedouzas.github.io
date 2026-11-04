# Exercise exr-nondegenerate-test: non-degeneracy is det != 0; of three symmetric matrices one is degenerate, and its
# kernel is the set of vectors pairing to zero with everything.
mats = {"diag(1, -1, -1, -1)": diagonal_matrix(QQ, [1, -1, -1, -1]), "[[1, 2], [2, 4]]": matrix(QQ, [[1, 2], [2, 4]]),
        "[[2, 1, 0], [1, 2, 1], [0, 1, 2]]": matrix(QQ, [[2, 1, 0], [1, 2, 1], [0, 1, 2]])}
for name, B in mats.items():
    print(name, ": det =", B.det(), "; non-degenerate:", B.det() != 0, ("; kernel basis " + str(B.right_kernel().basis())) if B.det() == 0 else "")
S = mats["[[1, 2], [2, 4]]"]; k = S.right_kernel().basis()[0]
print("the kernel vector", k, "pairs to zero with e1 and e2:", [k * S * e for e in identity_matrix(QQ, 2).rows()])

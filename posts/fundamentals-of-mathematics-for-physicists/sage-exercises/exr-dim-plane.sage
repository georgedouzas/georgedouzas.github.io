# Exercise exr-dim-plane: the plane x + y + z = 0 in Q^3 is a subspace of dimension 2 (the kernel of the row (1, 1, 1));
# Sage returns an echelonised basis, and the two vectors (1, -1, 0), (0, 1, -1) are another basis of it.
P = matrix(QQ, [[1, 1, 1]]).right_kernel()
print("the plane x + y + z = 0:", P)
print("dimension:", P.dimension())
alt = [vector(QQ, [1, -1, 0]), vector(QQ, [0, 1, -1])]
print("(1, -1, 0), (0, 1, -1) lie in the plane and span it:", all(v in P for v in alt) and P.span(alt) == P)

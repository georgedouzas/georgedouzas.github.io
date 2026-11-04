# Exercise exr-lrep-cartan-b2-g2: Cartan matrices A_ij = 2(alpha_i . alpha_j)/(alpha_j . alpha_j) computed from the
# simple roots in Sage's ambient space, compared with CartanMatrix (Sage's convention is the transpose), and the
# products A_ij A_ji that count the lines of the Dynkin diagram.
for ct in [["A", 2], ["B", 2], ["G", 2], ["A", 3]]:
    L = RootSystem(ct).ambient_space(); a = [x.to_vector() for x in L.simple_roots()]
    r = len(a)
    A = matrix(QQ, r, r, lambda i, j: 2 * a[i].dot_product(a[j]) / a[j].dot_product(a[j]))
    print(ct, "simple roots", [tuple(x) for x in a]); print(A)
    print("   equals the transpose of Sage's CartanMatrix:", A == CartanMatrix(ct).transpose(),
          "; A_12 A_21 =", A[0, 1] * A[1, 0], "; det =", A.det())
    print(CartanType(ct).dynkin_diagram())

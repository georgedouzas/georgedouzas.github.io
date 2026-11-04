# Exercise exr-dual-basis-general: a function returning the dual basis of any basis of Q^n (rows of the inverse of the
# matrix of columns), tested on the standard basis, on f = ((2, 1), (1, 2)) and on a basis of Q^3.
def dual_basis(vectors):
    B = matrix(QQ, vectors).transpose()
    return B.inverse().rows()
for name, b in [("standard", identity_matrix(QQ, 2).rows()), ("f", [vector(QQ, [2, 1]), vector(QQ, [1, 2])]),
                ("b", [vector(QQ, [1, 1, 0]), vector(QQ, [0, 1, 1]), vector(QQ, [1, 0, 1])])]:
    d = dual_basis(b)
    print(name, ": dual basis", d, "; d^i(b_j) = delta:", matrix(QQ, d) * matrix(QQ, b).transpose() == 1)

# Exercise exr-signature-3x3: the signature of a symmetric 3 x 3 matrix from the signs of its eigenvalues and from
# Sage's QuadraticForm; B = [[2, 1, 0], [1, 2, 1], [0, 1, 2]] is positive definite, C = [[0, 1, 0], [1, 0, 0], [0, 0, -1]]
# has signature (1, 2, 0).
for name, B in [("B", matrix(QQ, [[2, 1, 0], [1, 2, 1], [0, 1, 2]])), ("C", matrix(QQ, [[0, 1, 0], [1, 0, 0], [0, 0, -1]]))]:
    ev = B.change_ring(QQbar).eigenvalues()
    signs = (sum(1 for e in ev if e > 0), sum(1 for e in ev if e < 0), sum(1 for e in ev if e == 0))
    print(name, ": eigenvalues", [e.radical_expression() for e in ev], "; signs (+, -, 0) =", signs, "; QuadraticForm signature:", QuadraticForm(QQ, 2 * B).signature_vector())

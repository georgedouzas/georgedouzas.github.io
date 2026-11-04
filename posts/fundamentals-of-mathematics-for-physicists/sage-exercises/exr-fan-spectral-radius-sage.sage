# Exercise exr-fan-spectral-radius-sage: Gelfand's formula on two matrices and the normalised power iteration on the
# matrix of Izaac and Wang, Problem P7.6.
A = matrix(QQ, [[1/2, 4], [0, 1/2]])
B = matrix(QQ, [[0, 9, 0], [1, 0, 0], [0, 2, 2]])
print("eigenvalues of A:", A.eigenvalues(), "  of B:", B.eigenvalues(), "  char poly of B:", B.charpoly().factor())
for n in [1, 10, 100]:
    print("n = %3d   ||A^n||^(1/n) = %.6f   ||B^n||^(1/n) = %.6f" % (n, (A ^ n).change_ring(RDF).norm(2) ^ (1 / n), (B ^ n).change_ring(RDF).norm(2) ^ (1 / n)))
x = vector(RDF, [1, 1, 1]) / sqrt(3.0)
Bf = B.change_ring(RDF)
for k in range(1, 6):
    y = Bf * x
    lam = x.dot_product(y)
    x = y / y.norm()
    print("step %d: x = (%.4f, %.4f, %.4f), Rayleigh quotient %.4f" % (k, x[0], x[1], x[2], lam))

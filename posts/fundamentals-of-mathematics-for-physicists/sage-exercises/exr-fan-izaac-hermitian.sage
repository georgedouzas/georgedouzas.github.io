# Exercise exr-fan-izaac-hermitian: the Hermitian matrix of Izaac and Wang, Problem P7.1: eigenvalues, norm, spectral
# radius and the unitary e^{-iA}.
A = matrix(CDF, [[4, -I, 2], [I, 2, 2 + 7 * I], [2, 2 - 7 * I, -2]])
print("Hermitian:", (A - A.conjugate_transpose()).norm() < 1e-12)
ev = sorted([e.real() for e in A.eigenvalues()])
print("eigenvalues:", ["%.6f" % e for e in ev])
print("||A|| = %.6f   r(A) = %.6f" % (A.norm(2), max(abs(e) for e in ev)))
D, P = A.eigenmatrix_right()
U = P * diagonal_matrix(CDF, [exp(-I * d) for d in D.diagonal()]) * P.inverse()
print("||e^{-iA}|| = %.6f   unitary: %s" % (U.norm(2), (U * U.conjugate_transpose() - identity_matrix(CDF, 3)).norm() < 1e-10))
print("power iteration finds %.6f, inverse power iteration finds %.6f" % (max(ev, key=abs), min(ev, key=abs)))

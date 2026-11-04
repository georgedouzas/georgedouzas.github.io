# Exercise exr-bo-norm-2x2: operator norms of [[1, 2], [3, 4]] and [[1, 1], [0, 1]] from the eigenvalues of A^T A, compared
# with Sage's norm(2); the second norm is the golden ratio.
for A in [matrix(QQbar, [[1, 2], [3, 4]]), matrix(QQbar, [[1, 1], [0, 1]])]:
    ev = sorted((A.transpose() * A).eigenvalues(), reverse=True)
    print("A =", A.rows(), ": eigenvalues of A^T A:", ev, "; sqrt of the largest:", sqrt(ev[0]), "=", sqrt(ev[0]).n(digits=8), "; norm(2):", A.norm(2))
B = matrix(QQbar, [[1, 1], [0, 1]])
print("||B|| = (1 + sqrt 5)/2:", B.norm(2) == QQbar((1 + sqrt(5)) / 2), "; ||A|| for the first matrix = sqrt((30 + sqrt 884)/2):", matrix(QQbar, [[1, 2], [3, 4]]).norm(2) == QQbar(sqrt((30 + sqrt(884)) / 2)))

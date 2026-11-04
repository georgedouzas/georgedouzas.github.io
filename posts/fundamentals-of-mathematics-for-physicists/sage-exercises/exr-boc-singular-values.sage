# Exercise exr-boc-singular-values: the singular values of A = [[2, 1], [0, 1]] exactly, their product |det A|, the norm
# of the inverse and the condition number.
A = matrix(QQbar, [[2, 1], [0, 1]])
sig = sorted([sqrt(e) for e in (A.transpose() * A).eigenvalues()], reverse=True)
print("singular values:", sig, "=", [x.n(digits=6) for x in sig], "; equal to (sqrt 5 +- 1)/sqrt 2:", sig == [QQbar((sqrt(5) + 1) / sqrt(2)), QQbar((sqrt(5) - 1) / sqrt(2))])
print("sigma_1 sigma_2 =", sig[0] * sig[1], "= |det A| =", abs(A.det()))
print("||A|| =", A.norm(2), "; ||A^-1|| =", A.inverse().norm(2), "= 1/sigma_2:", A.inverse().norm(2) == 1 / sig[1])
print("condition number ||A|| ||A^-1|| =", A.norm(2) * A.inverse().norm(2), "= (3 + sqrt 5)/2:", A.norm(2) * A.inverse().norm(2) == QQbar((3 + sqrt(5)) / 2))

# Exercise exr-fan-operator-norms-matrix: the norms of A = [[1,2],[0,1]] from l^1, l^2, l^infinity to themselves, and of A^(-1).
A = matrix(QQ, [[1, 2], [0, 1]])
print("eigenvalues of A^T A:", (A.transpose() * A).charpoly().roots(QQbar, multiplicities=False))
n2 = sqrt(max((A.transpose() * A).eigenvalues()))
print("|A|_2 =", n2.radical_expression() if hasattr(n2, 'radical_expression') else n2, "=", n2.n(digits=6))
print("|A|_1 (max column sum) =", max(sum(abs(A[i, j]) for i in range(2)) for j in range(2)), " |A|_inf (max row sum) =", max(sum(abs(A[i, j]) for j in range(2)) for i in range(2)))
Ai = A.inverse()
print("|A^-1|_2 =", sqrt(max((Ai.transpose() * Ai).eigenvalues())).n(digits=6), " 1 + sqrt(2) =", (1 + sqrt(2)).n(digits=6))
print("|A|_2^2 <= |A|_1 |A|_inf:", n2^2 <= 9, "(", (n2^2).n(digits=5), "<= 9 )")

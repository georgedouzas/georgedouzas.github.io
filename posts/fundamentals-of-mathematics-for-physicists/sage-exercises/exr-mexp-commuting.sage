# Exercise exr-mexp-commuting: exp(A + B) = exp(A) exp(B) holds for commuting matrices and fails for the non-commuting
# pair N = [[0, 1], [0, 0]], N^T.
N = matrix(QQ, [[0, 1], [0, 0]])
Nt = N.transpose()
print("N N^T - N^T N =", (N * Nt - Nt * N).list(), "(non-zero: they do not commute)")
lhs = (N + Nt).exp().apply_map(lambda e: e.simplify_full())
rhs = (N.exp() * Nt.exp()).apply_map(lambda e: e.simplify_full())
print("exp(N + N^T) =", lhs.list())
print("exp(N) exp(N^T) =", rhs.list())
print("equal:", bool((lhs - rhs).apply_map(lambda e: e.simplify_full()) == 0))
D1 = matrix(QQ, [[1, 0], [0, 2]]); D2 = matrix(QQ, [[3, 0], [0, -1]])
print("for the commuting diagonal matrices diag(1, 2), diag(3, -1) the identity holds:", bool((D1 + D2).exp() == D1.exp() * D2.exp()))

# Exercise exr-fan-neumann-matrix: the Neumann series of A = (1/2)[[0,1],[1,0]], its sum (1 - A)^(-1) and the error of
# the partial sums in the operator norm, compared with the bound 2^(1-N).
A = matrix(QQ, [[0, 1], [1, 0]]) / 2
R = (1 - A).inverse()
print("(1 - A)^(-1) =", R.list(), " A^2 =", (A^2).list())
for N in (2, 4, 8, 16, 17):
    S = sum(A^k for k in range(N))
    err = (R - S).change_ring(RDF).norm(2)
    print("N = %2d: |(1-A)^(-1) - S_N| = %.3e, bound 2^(1-N) = %.3e" % (N, err, 2.0^(1 - N)))

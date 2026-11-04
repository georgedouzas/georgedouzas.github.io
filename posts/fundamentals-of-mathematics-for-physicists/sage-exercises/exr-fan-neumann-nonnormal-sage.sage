# Exercise exr-fan-neumann-nonnormal-sage: Neumann series of A = [[1/2, 4], [0, 1/2]], whose norm exceeds 1.
A = matrix(QQ, [[1/2, 4], [0, 1/2]])
inv = (1 - A).inverse()
print("(1 - A)^{-1} =", inv.rows())
print("||A|| =", A.change_ring(RDF).norm(2))
for N in [5, 10, 20, 40]:
    P = sum(A ^ k for k in range(N))
    E = inv - P
    print("N = %2d   partial sum (0,1) entry = %s   ||error|| = %.6g   8 (N+1) 2^(1-N) = %.6g" % (N, P[0, 1], E.change_ring(RDF).norm(2), 8 * (N + 1) * 2 ^ (1 - N)))
first = min(N for N in range(1, 100) if (inv - sum(A ^ k for k in range(N))).change_ring(RDF).norm(2) < 1)
print("first N with error below 1:", first)

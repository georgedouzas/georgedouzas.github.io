# Exercise exr-fd-sage-det-exp: det e^A = e^{tr A} for A = [[1, 2], [3, 4]], det(1 + A) as the product of 1 + lambda_k,
# and the expansion log det(1 + zA) = sum (-1)^(m+1) z^m tr(A^m)/m to order z^4.
A = matrix(SR, [[1, 2], [3, 4]])
print("det e^A =", A.exp().det().simplify_full(), " e^{tr A} = e^%s" % A.trace())
ev = A.eigenvalues()
print("eigenvalues:", ev, " det(1 + A) =", (identity_matrix(2) + A).det(), " prod(1 + lambda) =", prod(1 + l for l in ev).expand())
z = var("z")
lhs = taylor(log((identity_matrix(2) + z * A).det()), z, 0, 4)
rhs = sum((-1) ** (m + 1) * z ** m * (A ** m).trace() / m for m in range(1, 5))
print("log det(1 + zA) =", lhs); print("trace series   =", rhs.expand()); print("agree:", bool((lhs - rhs).expand() == 0))

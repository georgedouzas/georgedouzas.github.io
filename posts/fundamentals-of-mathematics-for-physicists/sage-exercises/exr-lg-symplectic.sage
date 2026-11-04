# Exercise exr-lg-symplectic: the symplectic group Sp(2n, R) = {M : M^T J M = J}. For n = 1, M^T J M = det(M) J, so
# Sp(2, R) = SL(2, R); for general n the condition is n(2n - 1) independent equations and dim Sp(2n) = n(2n + 1).
a, b, c, d = var("a b c d")
J = matrix(SR, [[0, 1], [-1, 0]])
M = matrix(SR, [[a, b], [c, d]])
print("M^T J M =", (M.transpose() * J * M).list(), "; equals det(M) J:", bool(M.transpose() * J * M == M.det() * J))
for n in [1, 2, 3]:
    Jn = block_matrix([[zero_matrix(QQ, n), identity_matrix(QQ, n)], [-identity_matrix(QQ, n), zero_matrix(QQ, n)]])
    # tangent space at 1: X^T J + J X = 0, a linear system for the (2n)^2 entries of X
    E = [matrix(QQ, 2 * n, 2 * n, {(i, j): 1}) for i in range(2 * n) for j in range(2 * n)]
    A = matrix(QQ, [(e.transpose() * Jn + Jn * e).list() for e in E]).transpose()
    print(f"n = {n}: dim sp({2*n}) = {4*n*n - A.rank()}, formula n(2n+1) = {n*(2*n+1)}")

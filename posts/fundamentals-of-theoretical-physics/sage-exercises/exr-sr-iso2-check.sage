# Exercise exr-sr-iso2-check: the little algebra of k = (1, 0, 0, 1) is spanned by J3, K1 - J2, K2 + J1, with the brackets
# of the Euclidean algebra iso(2).
eps = lambda i, j, k: sign(Permutation([i + 1, j + 1, k + 1]).signature()) if len({i, j, k}) == 3 else 0
J = [matrix(QQ, 4, 4, lambda a, b: -eps(i, a - 1, b - 1) if a > 0 and b > 0 else 0) for i in range(3)]
K = [matrix(QQ, 4, 4, lambda a, b: 1 if (a, b) in [(0, i + 1), (i + 1, 0)] else 0) for i in range(3)]
k = vector(QQ, [1, 0, 0, 1])
E1, E2, J3 = K[0] - J[1], K[1] + J[0], J[2]
print("J3 k, E1 k, E2 k:", J3*k, E1*k, E2*k)
br = lambda X, Y: X*Y - Y*X
print("[E1,E2] = 0:", br(E1, E2) == 0, " [J3,E1] = E2:", br(J3, E1) == E2, " [J3,E2] = -E1:", br(J3, E2) == -E1)
basis = J + K
A = matrix(QQ, [list(X*k) for X in basis]).transpose()     # columns: X k for the six basis elements
print("dimension of the stabiliser algebra:", A.right_kernel().dimension())

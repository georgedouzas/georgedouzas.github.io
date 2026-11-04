# Exercise exr-sr-ab-casimirs: the combinations A = (J + iK)/2, B = (J - iK)/2 in the four-vector representation; their
# brackets and the Casimirs sum A_i^2 = sum B_i^2 = -3/4.
eps = lambda i, j, k: sign(Permutation([i + 1, j + 1, k + 1]).signature()) if len({i, j, k}) == 3 else 0
J = [matrix(QQbar, 4, 4, lambda a, b: -eps(i, a - 1, b - 1) if a > 0 and b > 0 else 0) for i in range(3)]
K = [matrix(QQbar, 4, 4, lambda a, b: 1 if (a, b) in [(0, i + 1), (i + 1, 0)] else 0) for i in range(3)]
A = [(J[i] + I*K[i])/2 for i in range(3)]; B = [(J[i] - I*K[i])/2 for i in range(3)]
br = lambda X, Y: X*Y - Y*X
print("[A,A] = eps A:", all(br(A[i], A[j]) == sum(eps(i, j, k)*A[k] for k in range(3)) for i in range(3) for j in range(3)))
print("[B,B] = eps B:", all(br(B[i], B[j]) == sum(eps(i, j, k)*B[k] for k in range(3)) for i in range(3) for j in range(3)))
print("[A,B] = 0:", all(br(A[i], B[j]) == 0 for i in range(3) for j in range(3)))
print("sum A_i^2 ="); print(sum(a^2 for a in A)); print("sum B_i^2 ="); print(sum(b^2 for b in B))

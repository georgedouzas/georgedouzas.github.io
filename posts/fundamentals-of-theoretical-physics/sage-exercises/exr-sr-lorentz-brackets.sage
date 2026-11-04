# Exercise exr-sr-lorentz-brackets: the brackets of the rotation and boost generators J_i, K_i and of the M^{mu nu}.
eta = diagonal_matrix(QQ, [1, -1, -1, -1])
def M(m, n):
    return matrix(QQ, 4, 4, lambda a, b: eta[m, a]*(1 if n == b else 0) - eta[n, a]*(1 if m == b else 0))
eps = lambda i, j, k: sign(Permutation([i + 1, j + 1, k + 1]).signature()) if len({i, j, k}) == 3 else 0
J = [matrix(QQ, 4, 4, lambda a, b: -eps(i, a - 1, b - 1) if a > 0 and b > 0 else 0) for i in range(3)]
K = [matrix(QQ, 4, 4, lambda a, b: 1 if (a, b) in [(0, i + 1), (i + 1, 0)] else 0) for i in range(3)]
br = lambda X, Y: X*Y - Y*X
ok = all(br(J[i], J[j]) == sum(eps(i, j, k)*J[k] for k in range(3)) and br(J[i], K[j]) == sum(eps(i, j, k)*K[k] for k in range(3))
         and br(K[i], K[j]) == -sum(eps(i, j, k)*J[k] for k in range(3)) for i in range(3) for j in range(3))
print("[J,J] = eps J, [J,K] = eps K, [K,K] = -eps J:", ok)
okM = all(br(M(m, n), M(r, s)) == eta[n, r]*M(m, s) - eta[m, r]*M(n, s) - eta[n, s]*M(m, r) + eta[m, s]*M(n, r)
          for m in range(4) for n in range(4) for r in range(4) for s in range(4))
print("M^{mu nu} relations:", okM)
print("J_i = (1/2) eps_ijk M^{jk} and K_i = M^{0i}:", all(J[i] == sum(eps(i, j, k)*M(j + 1, k + 1) for j in range(3) for k in range(3))/2 and K[i] == M(0, i + 1) for i in range(3)))

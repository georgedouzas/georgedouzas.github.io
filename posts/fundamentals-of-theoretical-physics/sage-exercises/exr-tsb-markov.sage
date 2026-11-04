# Exercise exr-tsb-markov (Boudreau-Swanson, Ch. 7, Exercise 16): the three-state Markov chain with transition matrix P
# (rows sum to one); right and left eigenvectors, the stationary distribution w = w P, and the limit of P^n.
P = matrix(QQ, [[3/4, 1/4, 0], [0, 2/3, 1/3], [1/4, 1/4, 1/2]])
print("row sums:", [sum(r) for r in P.rows()])
print("eigenvalues:", P.charpoly().roots(QQbar, multiplicities=False))
print("right eigenvectors:", P.right_eigenvectors())
K = (P.transpose() - 1).right_kernel().basis()[0]; w = K / sum(K)
print("stationary distribution w:", w, "  w P = w:", w * P == w)
Pn = (P^60).n(digits=8)
print("P^60 =\n", Pn)
print("every row of P^n tends to w:", all(abs(Pn[i, j] - w[j]) < 1e-8 for i in range(3) for j in range(3)))
print("detailed balance w_i P_ij = w_j P_ji holds:", all(w[i] * P[i, j] == w[j] * P[j, i] for i in range(3) for j in range(3)))

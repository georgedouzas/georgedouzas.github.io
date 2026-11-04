# Exercise exr-qm-pt-anharmonic-sage: <n|x^4|n> from truncated ladder matrices, and exact levels of x^2/2 + p^2/2 + lambda x^4.
N = 12
a = matrix(QQbar, N, N, lambda i, j: QQbar(j).sqrt() if i == j - 1 else 0)
X = (a + a.transpose()) / QQbar(2).sqrt()
X4 = X ** 4
print("<n|x^4|n>, n = 0..11:", [QQ(X4[n, n]) for n in range(N)])
print("(3/4)(2n^2 + 2n + 1):  ", [QQ(3) / 4 * (2 * n ** 2 + 2 * n + 1) for n in range(N)])
NB = 60; lam = 0.1
an = matrix(RDF, NB, NB, lambda i, j: sqrt(j) if i == j - 1 else 0); Xn = (an + an.transpose()) / sqrt(2.0)
H = diagonal_matrix(RDF, [n + 0.5 for n in range(NB)]) + lam * Xn ** 4
E = sorted(H.eigenvalues())[:3]
print("lambda = 0.1: exact", [round(e, 4) for e in E], " first order", [n + 0.5 + 0.75 * lam * (2 * n ** 2 + 2 * n + 1) for n in range(3)])
NB2 = 80; an2 = matrix(RDF, NB2, NB2, lambda i, j: sqrt(j) if i == j - 1 else 0); Xn2 = (an2 + an2.transpose()) / sqrt(2.0)
E2 = sorted((diagonal_matrix(RDF, [n + 0.5 for n in range(NB2)]) + lam * Xn2 ** 4).eigenvalues())[:3]
print("size 80 changes them by", max(abs(x - y) for x, y in zip(E, E2)))

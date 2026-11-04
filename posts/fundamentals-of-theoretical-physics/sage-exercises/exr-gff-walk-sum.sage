# Exercise exr-gff-walk-sum (Bauerschmidt-Brydges-Slade, Lemma 1.5.3, eq. 1.5.21): on a 6 x 6 torus with m^2 = 1/2 the
# walk sum sum_n (A^n)_{0x}/(4 + m^2)^{n+1}, A the adjacency matrix, against (-Delta + m^2)^{-1}; the walk counts
# (A^n)_{0x} and the probabilities p_n = A^n/4^n, which sum to one over x, so that chi = 1/m^2.
n_side = 6; V = [(i, j) for i in range(n_side) for j in range(n_side)]
idx = {v: k for k, v in enumerate(V)}
A = matrix(QQ, len(V), len(V))
for (i, j) in V:
    for (a, b) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        A[idx[(i, j)], idx[((i + a) % n_side, (j + b) % n_side)]] += 1
m2 = QQ(1) / 2
C = ((4 + m2) * identity_matrix(QQ, len(V)) - A).inverse()
Ad = A.change_ring(RDF); S = zero_matrix(RDF, len(V)); P = identity_matrix(RDF, len(V))
for n in range(400):
    S += P / (4 + 0.5)^(n + 1); P = P * Ad
print("max |walk sum - (-Delta + m^2)^{-1}| =", (S - C.change_ring(RDF)).norm(Infinity))
An = A^4
print("number of 4-step walks from (0,0) to (0,0), (1,1), (2,0):", An[0, idx[(0, 0)]], An[0, idx[(1, 1)]], An[0, idx[(2, 0)]])
print("sum_x p_4(0, x) =", sum(An[0, k] for k in range(len(V))) / 4^4)
print("sum_x C_0x =", sum(C[0, k] for k in range(len(V))), " (1/m^2 = 2)")

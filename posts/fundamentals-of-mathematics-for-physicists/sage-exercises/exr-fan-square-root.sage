# Exercise exr-fan-square-root: sqrt(P) for P = [[5,4],[4,5]] by the spectral calculus, and |N| for N = [[0,1],[0,0]].
P = matrix(QQ, [[5, 4], [4, 5]])
D, Q = P.change_ring(QQbar).eigenmatrix_right()
S = (Q * D.apply_map(sqrt) * Q.inverse()).change_ring(QQ)
print("eigenvalues:", D.diagonal(), "; sqrt(P) =", S.list(), "; squared:", (S^2).list())
N = matrix(QQ, [[0, 1], [0, 0]])
NN = N.transpose() * N
print("N^* N =", NN.list(), "is a projection:", NN^2 == NN, "; so |N| = N^*N, norm", NN.change_ring(RDF).norm(2), "= |N| =", N.change_ring(RDF).norm(2))

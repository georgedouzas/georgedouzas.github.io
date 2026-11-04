# Exercise exr-lrep-gm-trace: the Gell-Mann matrices are Hermitian and traceless with tr(lambda_A lambda_B) = 2 delta_AB,
# and the generators T_A = -(i/2) lambda_A are anti-Hermitian with tr(T_A T_B) = -(1/2) delta_AB.
lam = [matrix(SR, [[0,1,0],[1,0,0],[0,0,0]]), matrix(SR, [[0,-I,0],[I,0,0],[0,0,0]]), matrix(SR, [[1,0,0],[0,-1,0],[0,0,0]]),
       matrix(SR, [[0,0,1],[0,0,0],[1,0,0]]), matrix(SR, [[0,0,-I],[0,0,0],[I,0,0]]), matrix(SR, [[0,0,0],[0,0,1],[0,1,0]]),
       matrix(SR, [[0,0,0],[0,0,-I],[0,I,0]]), matrix(SR, [[1,0,0],[0,1,0],[0,0,-2]])/sqrt(3)]
T = [-I / 2 * l for l in lam]
print("Hermitian:", all(bool(l.conjugate_transpose() == l) for l in lam), " traceless:", all(bool(l.trace() == 0) for l in lam))
G = matrix(SR, 8, 8, lambda A, B: (lam[A] * lam[B]).trace()).simplify_full()
print("tr(lambda_A lambda_B) = 2 delta_AB:", bool(G == 2 * identity_matrix(8)))
print("T_A anti-Hermitian:", all(bool(t.conjugate_transpose() == -t) for t in T),
      " tr(T_A T_B) = -1/2 delta_AB:", bool(matrix(SR, 8, 8, lambda A, B: (T[A] * T[B]).trace()).simplify_full() == -identity_matrix(8) / 2))

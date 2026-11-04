# Exercise exr-csd-embedding-gellmann: the matrix of ad tau on su(3) in the basis i lambda_1, ..., i lambda_8,
# its eigenvalues, and the norm of tau compared with T_3.
lam = [matrix(SR, [[0, 1, 0], [1, 0, 0], [0, 0, 0]]), matrix(SR, [[0, -I, 0], [I, 0, 0], [0, 0, 0]]), matrix(SR, [[1, 0, 0], [0, -1, 0], [0, 0, 0]]),
       matrix(SR, [[0, 0, 1], [0, 0, 0], [1, 0, 0]]), matrix(SR, [[0, 0, -I], [0, 0, 0], [I, 0, 0]]), matrix(SR, [[0, 0, 0], [0, 0, 1], [0, 1, 0]]),
       matrix(SR, [[0, 0, 0], [0, 0, -I], [0, I, 0]]), matrix(SR, [[1, 0, 0], [0, 1, 0], [0, 0, -2]]) / sqrt(3)]
B = [I * l for l in lam]
tau = I / 3 * diagonal_matrix(SR, [1, 1, -2]); ad = lambda A, C: A * C - C * A
# coordinates: X = sum c_k (i lambda_k) with c_k = -Tr(X i lambda_k)/2
coord = lambda X: [(-(X * b).trace() / 2).simplify_full() for b in B]
M = matrix(SR, [coord(ad(tau, b)) for b in B]).transpose()
print("ad tau in the basis i lambda_k:"); print(M)
print("eigenvalues:", M.eigenvalues())
print("[PASS] ad tau rotates (i lambda_4, i lambda_5) and (i lambda_6, i lambda_7) and kills i lambda_1,2,3,8; eigenvalues 0 (x4), +i (x2), -i (x2):",
      sorted(M.eigenvalues(), key=str) == sorted([0, 0, 0, 0, I, I, -I, -I], key=str) and all(M.column(k).is_zero() for k in (0, 1, 2, 7)))
T3 = I / 2 * lam[2]
print("Tr tau^2 =", (tau * tau).trace(), ", Tr T_3^2 =", (T3 * T3).trace())
print("[PASS] tau = (2/sqrt 3) (i lambda_8 / 2) and Tr tau^2 = -2/3 = (4/3) Tr T_3^2:", bool((tau - 2 / sqrt(3) * I * lam[7] / 2).simplify_full().is_zero()) and bool((tau * tau).trace() == -2/3) and bool((tau * tau).trace() == 4/3 * (T3 * T3).trace()))

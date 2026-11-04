# Exercise exr-ssg-oscillator-truncation: on the truncated oscillator (N bosonic levels) the relation {Q, Q^dag} = H fails
# only on the top state |N-1, 1>: there Q would create an N-th bosonic quantum which the truncation removes. The
# defect {Q, Q^dag} - H is -N omega on that one state and zero elsewhere.
for N in (4, 6, 8):
    a = matrix(SR, N, N, lambda i, j: sqrt(j) if j == i + 1 else 0)
    b = matrix(SR, [[0, 1], [0, 0]])
    A = a.tensor_product(identity_matrix(2)); Bf = identity_matrix(N).tensor_product(b)
    H = A.transpose() * A + Bf.transpose() * Bf; Q = A.transpose() * Bf; Qd = Q.transpose()
    D = Q * Qd + Qd * Q - H
    nz = [(i, D[i, i]) for i in range(2 * N) if D[i, i] != 0]
    print("N =", N, ": non-zero entries of {Q, Q^dag} - H at (state index, value):", nz, "; off-diagonal zero:", (D - diagonal_matrix(D.diagonal())).is_zero())
    print("[PASS] N = %d: the defect is -N on the top fermionic state |N-1, 1> only:" % N, nz == [(2 * N - 1, -N)])

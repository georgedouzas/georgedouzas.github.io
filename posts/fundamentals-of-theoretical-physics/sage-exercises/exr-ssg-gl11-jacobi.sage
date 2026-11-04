# Exercise exr-ssg-gl11-jacobi: the Lie superalgebra gl(1|1) of 2x2 supermatrices. Even generators are the diagonal
# matrices, odd ones the off-diagonal; the bracket is [X, Y} = XY - (-1)^{|X||Y|} YX. We check that the bracket closes,
# that the odd generators Q = E_12, Qbar = E_21 satisfy {Q, Qbar} = 1 (a central element, the analogue of H), Q^2 = 0,
# and the graded Jacobi identity on all triples of basis elements.
E11 = matrix(QQ, [[1, 0], [0, 0]]); E22 = matrix(QQ, [[0, 0], [0, 1]]); Q = matrix(QQ, [[0, 1], [0, 0]]); Qb = Q.transpose()
basis = [(E11, 0), (E22, 0), (Q, 1), (Qb, 1)]
def br(X, Y): return X[0] * Y[0] - (-1)^(X[1] * Y[1]) * Y[0] * X[0]
print("[PASS] {Q, Qbar} = identity (central), Q^2 = Qbar^2 = 0:", br((Q, 1), (Qb, 1)) == identity_matrix(2) and (Q * Q).is_zero() and (Qb * Qb).is_zero())
print("[PASS] [E11 - E22, Q] = 2Q and [E11 - E22, Qbar] = -2Qbar (the even part grades the odd one):",
      br((E11 - E22, 0), (Q, 1)) == 2 * Q and br((E11 - E22, 0), (Qb, 1)) == -2 * Qb)
def jac(X, Y, Z):
    s = lambda A, B: (-1)^(A[1] * B[1])
    t1 = s(X, Z) * br(X, (br(Y, Z), (Y[1] + Z[1]) % 2))
    t2 = s(Y, X) * br(Y, (br(Z, X), (Z[1] + X[1]) % 2))
    t3 = s(Z, Y) * br(Z, (br(X, Y), (X[1] + Y[1]) % 2))
    return (t1 + t2 + t3).is_zero()
print("[PASS] graded Jacobi identity (-1)^{|X||Z|}[X,[Y,Z}} + cyclic = 0 on all 64 triples:", all(jac(X, Y, Z) for X in basis for Y in basis for Z in basis))
print("[PASS] with the commutator instead, [Q, Qbar] = E11 - E22 is not central: the odd generators need the anticommutator:",
      Q * Qb - Qb * Q == E11 - E22)

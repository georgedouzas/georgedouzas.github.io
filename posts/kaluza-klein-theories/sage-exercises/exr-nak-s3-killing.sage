# Exercise exr-nak-s3-killing: the six Killing fields of the round S^3 in R^4 are the rotation fields with matrices
# (M_ij)_{kl} = delta_ik delta_jl - delta_il delta_jk (the field M x). The combinations J^(+-)_1 = (M_23 +- M_14)/2,
# J^(+-)_2 = (M_31 +- M_24)/2, J^(+-)_3 = (M_12 +- M_34)/2 split so(4) into two commuting copies of su(2):
# so(4) = su(2) + su(2), the left and right actions of SU(2) on S^3 = SU(2). The six fields are linearly independent.
d = lambda a, b: 1 if a == b else 0
Mm = lambda i, j: matrix(QQ, 4, 4, lambda k, l: d(i, k) * d(j, l) - d(i, l) * d(j, k))
Jp = [(Mm(1, 2) + Mm(0, 3)) / 2, (Mm(2, 0) + Mm(1, 3)) / 2, (Mm(0, 1) + Mm(2, 3)) / 2]
Jm = [(Mm(1, 2) - Mm(0, 3)) / 2, (Mm(2, 0) - Mm(1, 3)) / 2, (Mm(0, 1) - Mm(2, 3)) / 2]
def lc(a, b, c): return Permutation([a + 1, b + 1, c + 1]).sign() if len({a, b, c}) == 3 else 0
com = lambda P, Q: P * Q - Q * P
print("[J+_1, J+_2] =", com(Jp[0], Jp[1]) == Jp[2] and "J+_3" or com(Jp[0], Jp[1]) == -Jp[2] and "-J+_3")
print("[PASS] the matrices J+_a and J-_a commute with each other:", all(com(Jp[a], Jm[b]) == 0 for a in range(3) for b in range(3)))
s = 1 if com(Jp[0], Jp[1]) == Jp[2] else -1
print("[PASS] each set closes into su(2): [J_a, J_b] = %s eps_abc J_c for both:" % ("+" if s == 1 else "-"), all(com(J[a], J[b]) == s * sum(lc(a, b, c) * J[c] for c in range(3)) for J in [Jp, Jm] for a in range(3) for b in range(3)))
x = vector(var("x0 x1 x2 x3"))
fields = [Mm(i, j) * x for i in range(4) for j in range(i + 1, 4)]
coeffs = matrix(QQ, [[f[k].coefficient(x[l]) for k in range(4) for l in range(4)] for f in fields])
print("[PASS] the six rotation fields are linearly independent (rank 6 = 3 * 4 / 2):", coeffs.rank() == 6)
print("[PASS] every field is tangent to S^3 (x . M x = 0):", all(bool((x * f).expand() == 0) for f in fields))

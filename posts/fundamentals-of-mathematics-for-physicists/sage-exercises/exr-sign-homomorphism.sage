# Exercise exr-sign-homomorphism: the sign of a permutation as a homomorphism S_3 -> {+1, -1} with Sage, its kernel
# A_3, and the quotient S_3 / A_3 of order 2.
S3 = SymmetricGroup(3)
print("signs:", {str(g): g.sign() for g in S3})
print("homomorphism property sign(gh) = sign(g) sign(h):", all((g * h).sign() == g.sign() * h.sign() for g in S3 for h in S3))
K = [g for g in S3 if g.sign() == 1]
A3 = AlternatingGroup(3)
print("kernel:", K, "= A_3:", set(K) == set(A3.list()))
Q = S3.quotient(S3.subgroup(K))
print("the quotient S_3 / A_3 has order", Q.order(), "and is cyclic:", Q.is_cyclic())

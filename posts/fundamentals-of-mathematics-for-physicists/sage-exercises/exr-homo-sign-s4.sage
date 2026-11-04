# Exercise exr-homo-sign-s4: the sign on S_4 equals the determinant of the permutation matrix, is a homomorphism on
# all 576 pairs, and has the alternating group A_4 as kernel.
S4 = SymmetricGroup(4)
print("sign = det of the permutation matrix:", all(g.sign() == g.matrix().det() for g in S4))
print("sign(gh) = sign(g) sign(h) on all", S4.order()**2, "pairs:", all((g * h).sign() == g.sign() * h.sign() for g in S4 for h in S4))
K = S4.subgroup([g for g in S4 if g.sign() == 1])
print("kernel has order", K.order(), "; equals A_4:", K.is_isomorphic(AlternatingGroup(4)) and K.order() == 12)
print("sign of a k-cycle is (-1)^(k-1):", [(k, S4(tuple(range(1, k + 1))).sign()) for k in (2, 3, 4)])

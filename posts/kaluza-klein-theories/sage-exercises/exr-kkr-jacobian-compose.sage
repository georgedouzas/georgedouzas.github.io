# Exercise exr-kkr-jacobian-compose: the Jacobians of circle reparametrisations form a group isomorphic to the
# additive group of functions eps: J(eps1) J(eps2) = J(eps1 + eps2), J(0) = 1, J(eps)^{-1} = J(-eps), det J = 1.
e1 = vector(SR, var("a0 a1 a2 a3")); e2 = vector(SR, var("b0 b1 b2 b3"))       # the gradients d eps1, d eps2
def J(de):
    M = identity_matrix(SR, 5)
    for i in range(4): M[4, i] = -de[i]
    return M
print("[PASS] J(eps1) J(eps2) = J(eps1 + eps2):", J(e1) * J(e2) == J(e1 + e2))
print("[PASS] J(eps)^{-1} = J(-eps) and det J = 1:", J(e1).inverse() == J(-e1) and bool(J(e1).det() == 1))
print("[PASS] the product is commutative (an abelian group):", J(e1) * J(e2) == J(e2) * J(e1))

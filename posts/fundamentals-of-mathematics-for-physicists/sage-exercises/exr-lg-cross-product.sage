# Exercise exr-lg-cross-product: R^3 with the cross product is a Lie algebra: the bracket is bilinear and
# antisymmetric, and the Jacobi identity follows from a x (b x c) = b (a.c) - c (a.b); the basis vectors satisfy
# e_a x e_b = eps_abc e_c, the structure constants of so(3).
A = vector(SR, var("a1 a2 a3")); B = vector(SR, var("b1 b2 b3")); C = vector(SR, var("c1 c2 c3"))
print("a x b + b x a =", A.cross_product(B) + B.cross_product(A))
print("a x (b x c) - [b (a.c) - c (a.b)] =", (A.cross_product(B.cross_product(C)) - (B * A.dot_product(C) - C * A.dot_product(B))).apply_map(expand))
jac = A.cross_product(B.cross_product(C)) + B.cross_product(C.cross_product(A)) + C.cross_product(A.cross_product(B))
print("Jacobi sum =", jac.apply_map(expand))
e = identity_matrix(QQ, 3).rows()
print("e1 x e2, e2 x e3, e3 x e1 =", e[0].cross_product(e[1]), e[1].cross_product(e[2]), e[2].cross_product(e[0]))
print("associative? (e1 x e1) x e2 =", e[0].cross_product(e[0]).cross_product(e[1]), "but e1 x (e1 x e2) =", e[0].cross_product(e[0].cross_product(e[1])))

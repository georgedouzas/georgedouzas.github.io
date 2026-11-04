# Exercise exr-sr-field-general: the tensor law F' = L F L^T with the passive boost B(-v) against the vector formula for
# the transformed fields, for symbolic E, B and a velocity in a general direction.
E = vector(SR, var('E1 E2 E3')); Bv = vector(SR, var('B1 B2 B3'))
eps = lambda i, j, k: sign(Permutation([i + 1, j + 1, k + 1]).signature()) if len({i, j, k}) == 3 else 0
F = matrix(SR, 4, 4)
for i in range(3):
    F[i + 1, 0] = E[i]; F[0, i + 1] = -E[i]
    for j in range(3):
        F[i + 1, j + 1] = -sum(eps(i, j, k)*Bv[k] for k in range(3))
v = vector(QQ, [3/13, 4/13, 12/13])*(1/2)       # |v| = 1/2
g = 1/sqrt(1 - v*v); vh = v/v.norm()
L = identity_matrix(SR, 4); L[0, 0] = g
for i in range(3):
    L[0, i + 1] = -g*v[i]; L[i + 1, 0] = -g*v[i]
    for j in range(3):
        L[i + 1, j + 1] = (1 if i == j else 0) + (g - 1)*vh[i]*vh[j]
Fp = L*F*L.transpose()
Ep = vector([Fp[i + 1, 0] for i in range(3)]); Bp = vector([-Fp[2, 3], -Fp[3, 1], -Fp[1, 2]])
par = lambda X: (X*vh)*vh
Ef = par(E) + g*(E + v.cross_product(Bv) - par(E + v.cross_product(Bv)))
Bf = par(Bv) + g*(Bv - v.cross_product(E) - par(Bv - v.cross_product(E)))
print("E' - formula:", (Ep - Ef).simplify_full(), "   B' - formula:", (Bp - Bf).simplify_full())

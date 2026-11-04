# Exercise exr-cs-invariant-inner-sphere: invariant inner products. On T_o S^2 = R^2 the symmetric matrices g with
# X^T g + g X = 0 for the generator X of SO(2) are the multiples of the identity, and these are invariant under the
# finite rotations too; on R^3 with the three generators of the vector representation of SO(3) (the isotropy
# representation of S^3 = SO(4)/SO(3)) the same holds: one radius.
a, b, c, t = var("a b c t", domain="real")
X = matrix(SR, [[0, -1], [1, 0]]); g = matrix(SR, [[a, b], [b, c]])
eqs = [e == 0 for e in (X.transpose() * g + g * X).list()]
print("solutions of X^T g + g X = 0:", solve(eqs, a, b, c))
R = matrix(SR, [[cos(t), -sin(t)], [sin(t), cos(t)]])
print("R(t)^T (a 1) R(t) = a 1:", bool((R.transpose() * a * identity_matrix(2) * R).simplify_full() == a * identity_matrix(2)))
gs = var("g11 g12 g13 g22 g23 g33", domain="real")
G = matrix(SR, [[gs[0], gs[1], gs[2]], [gs[1], gs[3], gs[4]], [gs[2], gs[4], gs[5]]])
L = [matrix(SR, [[0,0,0],[0,0,-1],[0,1,0]]), matrix(SR, [[0,0,1],[0,0,0],[-1,0,0]]), matrix(SR, [[0,-1,0],[1,0,0],[0,0,0]])]
eqs3 = [e == 0 for Lb in L for e in (Lb.transpose() * G + G * Lb).list()]
print("solutions for the vector representation of so(3):", solve(eqs3, *gs))

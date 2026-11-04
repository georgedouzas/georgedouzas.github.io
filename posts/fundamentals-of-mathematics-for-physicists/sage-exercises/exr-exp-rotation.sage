# Exercise exr-exp-rotation: the exponential of t L_3 is the rotation matrix about the third axis.
t = var('t', domain='real')
L3 = matrix(SR, [[0, -1, 0], [1, 0, 0], [0, 0, 0]])
print("L3^2 =", (L3^2).list(), "  L3^3 = -L3:", bool(L3^3 == -L3))
R = (t*L3).exp().apply_map(lambda e: e.rectform().simplify_full())
print("exp(t L3) ="); print(R)
print("rotation by t about the third axis:", bool(R == matrix(SR, [[cos(t), -sin(t), 0], [sin(t), cos(t), 0], [0, 0, 1]])))
print("orthogonal with determinant 1:", bool((R*R.transpose()).simplify_full() == identity_matrix(3)), bool(R.det().simplify_full() == 1))

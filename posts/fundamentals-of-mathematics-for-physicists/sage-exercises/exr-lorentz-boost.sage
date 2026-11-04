# Exercise exr-lorentz-boost: a boost along x with rapidity u preserves the Minkowski form, Lambda^T eta Lambda = eta,
# so it maps timelike vectors to timelike vectors and null vectors to null vectors.
u = var("u")
L = matrix(SR, [[cosh(u), sinh(u), 0, 0], [sinh(u), cosh(u), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
eta = diagonal_matrix(SR, [1, -1, -1, -1])
print("Lambda^T eta Lambda - eta =", (L.transpose() * eta * L - eta).simplify_full().list() == [0] * 16, "(all 16 entries vanish by cosh^2 - sinh^2 = 1)")
for v in [vector(SR, [1, 0, 0, 0]), vector(SR, [1, 1, 0, 0]), vector(SR, [2, 1, 0, 3])]:
    print("v =", v, ": eta(v, v) =", v * eta * v, "; eta(Lambda v, Lambda v) =", ((L * v) * eta * (L * v)).simplify_full())

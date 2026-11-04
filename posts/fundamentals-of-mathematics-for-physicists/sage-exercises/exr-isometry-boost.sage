# Exercise exr-isometry-boost: a Lorentz boost and a spatial rotation preserve the Minkowski metric, Lambda^T eta
# Lambda = eta, while a dilation rescales it; the pullback of a constant metric by a linear map is a matrix product.
eta = diagonal_matrix(SR, [1, -1, -1, -1])
xi, al, lam = var("xi al lam", domain="real")
boost = matrix(SR, [[cosh(xi), sinh(xi), 0, 0], [sinh(xi), cosh(xi), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
rot = matrix(SR, [[1, 0, 0, 0], [0, cos(al), -sin(al), 0], [0, sin(al), cos(al), 0], [0, 0, 0, 1]])
dil = lam * identity_matrix(SR, 4)
for name, L in [("boost by rapidity xi", boost), ("rotation by al about z", rot), ("dilation by lam", dil)]:
    pulled = (L.transpose() * eta * L).simplify_full()
    print(name, ": diagonal of Lambda^T eta Lambda =", pulled.diagonal(), "; isometry:", bool(pulled == eta))
v = vector(SR, [1, 0, 0, 0]); bv = (boost * v).simplify_full()
print("the boost sends the unit timelike vector (1, 0, 0, 0) to", bv, "with eta-norm", (bv * eta * bv).simplify_full(), " (still timelike, unit)")

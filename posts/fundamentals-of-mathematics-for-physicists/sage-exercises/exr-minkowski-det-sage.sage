# Exercise exr-minkowski-det-sage: determinants and signatures of the Minkowski metric, of a boosted Minkowski metric
# and of the five-dimensional product metric with a circle of radius R, by exact matrix computation.
eta = diagonal_matrix(QQ, [1, -1, -1, -1])
print("det eta =", eta.det(), "; signature (p, q, z) =", QuadraticForm(QQ, 2*eta).signature_vector())
xi = var("xi", domain="real")
L = matrix(SR, [[cosh(xi), sinh(xi), 0, 0], [sinh(xi), cosh(xi), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])   # boost along x
boosted = (L.transpose() * eta * L).simplify_full()
print("boost: Lambda^T eta Lambda == eta:", bool(boosted == eta), "; det Lambda =", L.det().simplify_full(), "; det of the boosted metric =", boosted.det())
R = var("R", domain="real"); assume(R > 0)
g5 = diagonal_matrix(SR, [1, -1, -1, -1, -R**2])
print("five-dimensional diag(1, -1, -1, -1, -R^2): det =", g5.det(), " (positive); sqrt|det| =", sqrt(g5.det()).canonicalize_radical())
print("its signature (p, q, z) =", QuadraticForm(QQ, 2*diagonal_matrix(QQ, [1, -1, -1, -1, -1])).signature_vector())

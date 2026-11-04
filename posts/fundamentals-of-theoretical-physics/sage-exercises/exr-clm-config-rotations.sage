# Exercise exr-clm-config-rotations: R = R_z(phi) R_x(theta) R_z(psi) is a rotation for all Euler angles, and the
# chart degenerates at theta = 0, where R depends on phi + psi only.
ph, th, ps = var("phi theta psi", domain="real")
def Rz(a): return matrix([[cos(a), -sin(a), 0], [sin(a), cos(a), 0], [0, 0, 1]])
def Rx(a): return matrix([[1, 0, 0], [0, cos(a), -sin(a)], [0, sin(a), cos(a)]])
R = Rz(ph) * Rx(th) * Rz(ps)
print("R^T R = 1:", bool((R.transpose() * R).simplify_full() == identity_matrix(3)), "; det R = 1:", bool(R.det().simplify_full() == 1))
print("R at theta = 0 equals R_z(phi + psi):", bool((R.subs(theta=0) - Rz(ph + ps)).simplify_full() == 0))
print("R at theta = 0, psi = -phi is the identity:", bool(R.subs(theta=0, psi=-ph).simplify_full() == identity_matrix(3)))

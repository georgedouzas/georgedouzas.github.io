# Exercise exr-goldstone-so3: the SO(3) vector model V = (lam/4)(phi.phi - a^2)^2. Find the vacuum, the mass matrix
# (Hessian of V at the vacuum) and its eigenvalues: one massive radial mode, m^2 = 2 lam a^2, and two zero modes, the
# Goldstone bosons tangent to the sphere of vacua SO(3)/SO(2) = S^2.
lam, a = var("lam a"); assume(lam > 0, a > 0)
p1, p2, p3 = var("p1 p2 p3"); phi = vector([p1, p2, p3])
V = lam/4 * (phi.dot_product(phi) - a^2)^2
vac = {p1: 0, p2: 0, p3: a}
print("V at the vacuum (0, 0, a):", V.subs(vac))
grad = vector([diff(V, p) for p in (p1, p2, p3)]).subs(vac)
print("gradient at the vacuum:", grad)
M = matrix(SR, 3, 3, lambda i, j: diff(V, [p1, p2, p3][i], [p1, p2, p3][j])).subs(vac)
print("mass matrix (Hessian) at the vacuum:"); print(M)
print("eigenvalues:", M.eigenvalues())
print("[PASS] two zero modes (Goldstone bosons) and one massive mode with m^2 = 2 lam a^2:", sorted(M.eigenvalues(), key=lambda e: bool(e != 0)) == [0, 0, 2*lam*a^2])
# the unbroken generator: the rotation about the vacuum direction annihilates the vacuum, the other two do not
L1 = matrix([[0, 0, 0], [0, 0, -1], [0, 1, 0]]); L2 = matrix([[0, 0, 1], [0, 0, 0], [-1, 0, 0]]); L3 = matrix([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
v0 = vector([0, 0, a])
print("L_a phi_0 for a = 1, 2, 3:", L1*v0, L2*v0, L3*v0)
print("[PASS] only L_3 (rotation about the vacuum axis) fixes the vacuum, so H = SO(2) and there are 3 - 1 = 2 broken generators:", (L1*v0 != 0) and (L2*v0 != 0) and (L3*v0 == 0))
print("[PASS] the zero modes of the mass matrix are the broken directions L_1 phi_0, L_2 phi_0:", (M*(L1*v0)).is_zero() and (M*(L2*v0)).is_zero())

# Exercise exr-lg-so3-commutator: the group commutator of small rotations about X and Y,
# R_1(e) R_2(e) R_1(-e) R_2(-e) = 1 + e^2 L_3 + O(e^3): the bracket [L_1, L_2] = L_3 measured by rotations.
e = var("e")
L = [matrix(SR, [[0, 0, 0], [0, 0, -1], [0, 1, 0]]), matrix(SR, [[0, 0, 1], [0, 0, 0], [-1, 0, 0]]), matrix(SR, [[0, -1, 0], [1, 0, 0], [0, 0, 0]])]
def R(a, t): return (t * L[a]).exp().apply_map(lambda x: x.rectform().simplify_full())
C = R(0, e) * R(1, e) * R(0, -e) * R(1, -e)
Cs = C.apply_map(lambda x: x.taylor(e, 0, 2))
print("commutator to order e^2 =", Cs.list())
print("equals 1 + e^2 L_3:", bool(Cs == identity_matrix(SR, 3) + e^2 * L[2]))
print("swapping the axes gives 1 - e^2 L_3:", bool((R(1, e) * R(0, e) * R(1, -e) * R(0, -e)).apply_map(lambda x: x.taylor(e, 0, 2)) == identity_matrix(SR, 3) - e^2 * L[2]))

# Exercise exr-fbun-constant-field-three: the constant su(2) potential A_x = c1 T_1, A_y = c2 T_2, A_z = c3 T_3 on
# R^3 has F_xy = c1 c2 T_3, F_yz = c2 c3 T_1, F_zx = c3 c1 T_2, and the invariant
# -2 sum_{i<j} tr(F_ij^2) = (c1^2 c2^2 + c2^2 c3^2 + c3^2 c1^2) / 2.
c1, c2, c3 = var("c1 c2 c3", domain="real")
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
T = [-I / 2 * sa for sa in s]
A = [c1 * T[0], c2 * T[1], c3 * T[2]]
F = {(i, j): A[i] * A[j] - A[j] * A[i] for i in range(3) for j in range(3)}   # derivative terms vanish
print("F_xy = c1 c2 T_3:", bool(F[0, 1] == c1 * c2 * T[2]))
print("F_yz = c2 c3 T_1:", bool(F[1, 2] == c2 * c3 * T[0]))
print("F_zx = c3 c1 T_2:", bool(F[2, 0] == c3 * c1 * T[1]))
inv = (-2 * sum((F[i, j] * F[i, j]).trace() for i in range(3) for j in range(i + 1, 3))).expand()
print("-2 sum_{i<j} tr(F_ij^2) =", inv)
print("vanishes only if at least two of the c_i vanish; at c = (1, 1, 0):", inv.subs(c1=1, c2=1, c3=0))

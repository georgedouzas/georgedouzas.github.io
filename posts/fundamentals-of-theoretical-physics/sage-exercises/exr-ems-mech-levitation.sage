# Exercise exr-ems-mech-levitation (Macchi et al., Problem 5.4): induced moment m = alpha B in B_z = B0 z/L.
s, z, B0, L, M, g, al = var('s z B0 L M g alpha'); assume(L > 0, M > 0, g > 0, B0 > 0)
Bs = -B0 * s / (2 * L); Bz = B0 * z / L
print("div B in cylindrical coordinates:", (diff(s * Bs, s) / s + diff(Bz, z)).simplify_full())
A = var('A', domain='positive')   # A = |alpha| for a diamagnet, alpha = -A
U = A / 2 * (Bs^2 + Bz^2) + M * g * z
eq = solve([diff(U, s) == 0, diff(U, z) == 0], [s, z])
print("equilibrium:", eq)
print("omega_z^2 =", (diff(U, z, 2) / M).simplify_full(), "; omega_s^2 =", (diff(U, s, 2) / M).simplify_full())
print("paramagnet (alpha > 0): U'' along z =", diff(-A / 2 * (Bs^2 + Bz^2), z, 2), "< 0, unstable")

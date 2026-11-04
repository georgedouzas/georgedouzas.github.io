# Exercise exr-ems-mech-coil-magnet (Pierrus, Question 4.29): force on a small magnet m e_z on the axis of a loop.
m, I, a, z = var('m I a z', domain='positive')
Bz = I * a^2 / (2 * (a^2 + z^2)^(3/2))
F = (m * diff(Bz, z)).simplify_full()
print("F_z =", F)
zmax = solve(diff(F, z) == 0, z)
print("extremum at", zmax)
Fmax = F.subs(z=a / 2).simplify_full()
print("F_z(a/2) =", Fmax, "; equals -24 m I/(25 sqrt5 a^2):", bool((Fmax + 24 * m * I / (25 * sqrt(5) * a^2)).simplify_full() == 0))

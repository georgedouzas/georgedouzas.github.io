# Exercise exr-qz-angular-momentum: the orbital angular momentum operators L_i = -i hbar eps_ijk x_j d_k acting on an
# arbitrary function f(x, y, z) satisfy [L_x, L_y] = i hbar L_z (and cyclic) and [L^2, L_z] = 0.
x, y, z, hbar = var("x y z hbar")
f = function("f")(x, y, z)
Lx = lambda g: -I * hbar * (y * diff(g, z) - z * diff(g, y))
Ly = lambda g: -I * hbar * (z * diff(g, x) - x * diff(g, z))
Lz = lambda g: -I * hbar * (x * diff(g, y) - y * diff(g, x))
comm = lambda A, B, g: (A(B(g)) - B(A(g))).expand()
print("[L_x, L_y] f = i hbar L_z f:", bool(comm(Lx, Ly, f) == (I * hbar * Lz(f)).expand()))
print("[L_y, L_z] f = i hbar L_x f:", bool(comm(Ly, Lz, f) == (I * hbar * Lx(f)).expand()))
print("[L_z, L_x] f = i hbar L_y f:", bool(comm(Lz, Lx, f) == (I * hbar * Ly(f)).expand()))
L2 = lambda g: Lx(Lx(g)) + Ly(Ly(g)) + Lz(Lz(g))
print("[L^2, L_z] f = 0:", bool(comm(L2, Lz, f) == 0))
print("example: L_z (x + i y) = hbar (x + i y):", bool(Lz(x + I * y) == hbar * (x + I * y)))

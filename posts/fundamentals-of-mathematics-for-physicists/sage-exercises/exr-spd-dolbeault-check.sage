# Exercise exr-spd-dolbeault-check: on C, i(sigma_1 d_x + sigma_2 d_y) = [[0, 2i d_z], [2i d_zbar, 0]] and its square is
# -4 d_z d_zbar = -Laplacian.
x, y = var("x y", domain="real")
f = function("f")(x, y); g = function("g")(x, y)
dz = lambda h: (diff(h, x) - I * diff(h, y)) / 2
dzb = lambda h: (diff(h, x) + I * diff(h, y)) / 2
s1 = matrix(SR, [[0, 1], [1, 0]]); s2 = matrix(SR, [[0, -I], [I, 0]])
psi = matrix(SR, 2, 1, [f, g])
D = lambda p: I * (s1 * p.apply_map(lambda h: diff(h, x)) + s2 * p.apply_map(lambda h: diff(h, y)))
Dz = lambda p: matrix(SR, 2, 1, [2 * I * dz(p[1, 0]), 2 * I * dzb(p[0, 0])])
print("D = [[0, 2i d_z], [2i d_zbar, 0]]:", bool((D(psi) - Dz(psi)).simplify_full() == 0))
print("D^2 psi + Laplacian psi =", (D(D(psi)) + psi.apply_map(lambda h: diff(h, x, 2) + diff(h, y, 2))).simplify_full().list())
print("-4 d_z d_zbar f + Laplacian f =", (-4 * dz(dzb(f)) + diff(f, x, 2) + diff(f, y, 2)).simplify_full())

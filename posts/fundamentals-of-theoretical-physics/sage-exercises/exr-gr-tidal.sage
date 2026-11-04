# Exercise exr-gr-tidal: the tidal tensor of a point mass. Two nearby freely falling particles separated by xi have
# relative acceleration xi''^i = -(d_i d_j Phi) xi^j; for Phi = -G M / r at (r, 0, 0) this is diag(2, -1, -1) G M / r^3:
# stretching along the radius, compression across it.
x, y, z, G, M, r = var('x y z G M r'); assume(r > 0)
Phi = -G*M/sqrt(x^2 + y^2 + z^2)
T = -matrix(SR, 3, 3, lambda i, j: diff(Phi, [x, y, z][i], [x, y, z][j])).subs(x=r, y=0, z=0).simplify_full()
print("-d_i d_j Phi at (r, 0, 0) =", T)
print("equals G M / r^3 diag(2, -1, -1):", bool(T == G*M/r^3*diagonal_matrix([2, -1, -1])), "; trace (Laplacian, zero in vacuum):", T.trace())
# numbers: an astronaut of height 2 m at 400 km above the Earth
GM, R = 3.986e14, 6.371e6 + 4.0e5
print("tidal stretch over 2 m at 400 km altitude: %.2e m/s^2" % (2*GM/R^3*2))

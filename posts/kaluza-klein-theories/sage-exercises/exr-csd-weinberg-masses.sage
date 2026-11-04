# Exercise exr-csd-weinberg-masses: the gauge boson mass matrix M^2_ij = -2 sum_a Tr([T_i, phi_a][T_j, phi_a]) at the
# vacuum Phi = (0, 1), in units of 1/R^2, for the basis T_1, T_2, T_3 = (i/2) sigma_a (upper block) and
# T_8 = (i/2) lambda_8, all normalised by Tr(T_i T_j) = -delta_ij/2. Eigenvalues, eigenvectors and the mixing angle.
E = lambda i, j: matrix(SR, 3, 3, lambda a, b: 1 if (a, b) == (i, j) else 0)
ad = lambda A, B: A * B - B * A
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
def up(m): M = zero_matrix(SR, 3, 3); M[0:2, 0:2] = m; return M
T = [I / 2 * up(m) for m in s] + [I / (2 * sqrt(3)) * diagonal_matrix(SR, [1, 1, -2])]
Pp = E(1, 2); Pm = -E(2, 1); f = [(Pp + Pm) / 2, I * (Pp - Pm) / 2]
M2 = matrix(SR, 4, 4, lambda i, j: -2 * sum((ad(T[i], x) * ad(T[j], x)).trace() for x in f))
print("M^2 ="); print(M2)
ev = M2.eigenvectors_right()
for val, vecs, mult in ev: print("eigenvalue", val, "eigenvectors", vecs)
th = arctan(sqrt(3))
Zdir = vector(SR, [0, 0, cos(th), -sin(th)]); Adir = vector(SR, [0, 0, sin(th), cos(th)])
print("[PASS] eigenvalues 1/2, 1/2 (W_1, W_2), 2 (Z = cos th T_3 - sin th T_8), 0 (photon = sin th T_3 + cos th T_8) with tan th = sqrt 3:",
      sorted(M2.eigenvalues()) == [0, 1/2, 1/2, 2] and bool((M2 * Zdir - 2 * Zdir).simplify_full().is_zero()) and bool((M2 * Adir).simplify_full().is_zero()))
print("[PASS] th = 60 degrees, sin^2 th = 3/4, and M_W^2 / M_Z^2 = 1/4 = cos^2 th:", bool(th == pi / 3) and bool(sin(th)^2 == 3/4) and bool((1/2) / 2 == cos(th)^2))

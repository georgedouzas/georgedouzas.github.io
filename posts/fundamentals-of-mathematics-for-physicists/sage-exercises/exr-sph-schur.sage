# Exercise exr-sph-schur: Schur orthogonality for the spin-1/2 Wigner matrix in Euler angles, and the vanishing of the
# integrals between spin 1/2 and spin 1.
th, ph, psi = var("theta phi psi")
def haar(f):
    return integrate(integrate(integrate(f * sin(th), psi, 0, 4 * pi), ph, 0, 2 * pi), th, 0, pi) / (16 * pi ** 2)
Dh = matrix(SR, [[exp(-I * (ph + psi) / 2) * cos(th / 2), -exp(-I * (ph - psi) / 2) * sin(th / 2)], [exp(I * (ph - psi) / 2) * sin(th / 2), exp(I * (ph + psi) / 2) * cos(th / 2)]])
d1 = matrix(SR, [[(1 + cos(th)) / 2, -sin(th) / sqrt(2), (1 - cos(th)) / 2], [sin(th) / sqrt(2), cos(th), -sin(th) / sqrt(2)], [(1 - cos(th)) / 2, sin(th) / sqrt(2), (1 + cos(th)) / 2]])
D1 = diagonal_matrix(SR, [exp(-I * ph), 1, exp(I * ph)]) * d1 * diagonal_matrix(SR, [exp(-I * psi), 1, exp(I * psi)])
vals = {}
for i in range(2):
    for j in range(2):
        for k in range(2):
            for l in range(2):
                vals[(i, j, k, l)] = haar((Dh[i, j] * conjugate(Dh[k, l])).simplify_full()).simplify_full()
print("non-zero integrals int D_ij conj(D_kl) dg for spin 1/2:", {key: v for key, v in vals.items() if v != 0})
print("all equal to delta_ik delta_jl / 2:", all(v == (QQ(1) / 2 if (i, j) == (k, l) else 0) for (i, j, k, l), v in vals.items()))
mixed = [haar((Dh[i, j] * conjugate(D1[k, l])).simplify_full()).simplify_full() for (i, j, k, l) in [(0, 0, 0, 0), (0, 0, 1, 1), (0, 1, 0, 1), (1, 0, 2, 1), (1, 1, 2, 2)]]
print("integrals between spin 1/2 and spin 1 for five index choices:", mixed)

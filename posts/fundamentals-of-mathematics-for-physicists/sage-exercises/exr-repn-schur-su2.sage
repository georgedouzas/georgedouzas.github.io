# Exercise exr-repn-schur-su2: complex 2 x 2 matrices commuting with the spinor generators T_a = -(i/2) sigma_a are
# the multiples of the identity, while those commuting with T_3 alone are the diagonal matrices.
K.<i> = QuadraticField(-1)
sig = [matrix(K, [[0, 1], [1, 0]]), matrix(K, [[0, -i], [i, 0]]), matrix(K, [[1, 0], [0, -1]])]
T = [-i / 2 * s for s in sig]
R = PolynomialRing(K, 4, 'x'); X = matrix(R, 2, 2, R.gens())
def commutant(gens):
    eqs = [e for A in gens for e in (X * A - A * X).list()]
    M = matrix(K, [[e.coefficient(v) for v in R.gens()] for e in eqs])
    return [matrix(K, 2, 2, list(b)) for b in M.right_kernel().basis()]
print("brackets [T_1, T_2] = T_3:", T[0] * T[1] - T[1] * T[0] == T[2])
print("commutant of T_1, T_2, T_3:", commutant(T))
print("commutant of T_3 alone:", commutant([T[2]]))

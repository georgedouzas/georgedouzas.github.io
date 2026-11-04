# Exercise exr-repn-schur-vector-solve: the 18 linear equations X L_a = L_a X (a = 1, 2) for a 3 x 3 matrix X and
# their solution: X is a multiple of the identity; commuting with L_3 alone leaves a 3-dimensional family.
L = [matrix(QQ, [[0,0,0],[0,0,-1],[0,1,0]]), matrix(QQ, [[0,0,1],[0,0,0],[-1,0,0]]), matrix(QQ, [[0,-1,0],[1,0,0],[0,0,0]])]
R = PolynomialRing(QQ, 9, 'x'); X = matrix(R, 3, 3, R.gens())
def solve_commutant(gens):
    eqs = [e for A in gens for e in (X * A - A * X).list()]
    M = matrix(QQ, [[e.coefficient(v) for v in R.gens()] for e in eqs])
    return len(eqs), M.right_kernel().basis()
n, basis = solve_commutant([L[0], L[1]])
print("equations from L_1 and L_2:", n, "; solution space:", [matrix(QQ, 3, 3, list(b)) for b in basis])
n3, basis3 = solve_commutant([L[2]])
print("commuting with L_3 only: dimension", len(basis3))
for b in basis3:
    print(matrix(QQ, 3, 3, list(b))); print()
print("[L_1, L_2] = L_3:", L[0] * L[1] - L[1] * L[0] == L[2])

# name: de-mexp-boost
# technique: closed forms of matrix exponentials, proved against mathlib's NormedSpace.exp (matrix_exponential recipe)
# goal: NormedSpace.exp (t • (0, 1; 1, 0)) = (cosh t, sinh t; sinh t, cosh t)
# description: A = P D P⁻¹ over ℚ; Matrix.exp_conj and Matrix.exp_diagonal give the entries, compared after unfolding cosh and sinh.
MATRIX = matrix(QQ, [[0, 1], [1, 0]])
PARAMETER = 't'
closed, proof = diagonalizable_exponential(MATRIX, PARAMETER)
print(closed)

# name: de-mexp-NT
# technique: closed forms of matrix exponentials, proved against mathlib's NormedSpace.exp (matrix_exponential recipe)
# goal: NormedSpace.exp Nᵀ = 1 + Nᵀ for Nᵀ = (0, 0; 1, 0)
# description: N² = 0, so the exponential series stops: exp N = 1 + N, from NormedSpace.exp_eq_tsum_rat.
MATRIX = matrix(QQ, [[0, 0], [1, 0]])
PARAMETER = 't'
closed, proof = nilpotent_exponential(MATRIX)
print(closed)

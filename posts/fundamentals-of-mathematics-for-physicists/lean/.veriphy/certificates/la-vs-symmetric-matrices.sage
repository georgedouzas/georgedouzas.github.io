# name: la-vs-symmetric-matrices
# technique: matrix computation read from the goal (matrix_from_goal recipe), with the exercise's Sage solution as oracle
# goal: identities, determinants of the explicit matrices of exr-vs-symmetric-matrices (symmetric A, B; rotations R1, R2)
# description: the oracle reruns the exercise (A, B symmetric, R1, R2 rotations, R1 + R2 not one); then Sage evaluates
#   lhs - rhs from the goal and Lean proves it entrywise with norm_num, or by the closed 3×3 det formula and ring.
import re
# oracle (sage/exercises/exr-vs-symmetric-matrices.sage)
A = matrix(QQ, [[1, 2, 0], [2, 3, 1], [0, 1, 4]]); B = matrix(QQ, [[0, 1, 1], [1, 0, 2], [1, 2, 5]])
assert A.is_symmetric() and B.is_symmetric() and (A + B).is_symmetric() and (3 * A).is_symmetric()
R1 = matrix(QQ, [[0, -1, 0], [1, 0, 0], [0, 0, 1]]); R2 = matrix(QQ, [[1, 0, 0], [0, 0, -1], [0, 1, 0]])
assert R1.transpose() * R1 == 1 and R1.det() == 1 and R2.transpose() * R2 == 1 and R2.det() == 1
assert (R1 + R2).transpose() * (R1 + R2) != 1
print('oracle: (R1 + R2)^T (R1 + R2) =', ((R1 + R2).transpose() * (R1 + R2)).rows())
# recipe core (matrix_from_goal)
e = goal.expr
assert e is not None, 'the goal has no symbolic form'
if hasattr(e, 'nrows'):
    assert e.simplify_full() == 0, f'the identity is false: lhs - rhs = {matrix_term(e.simplify_full(), problem)}'
    proof = tactic('ext i j\nfin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]')
else:
    assert bool(e.simplify_full() == 0), f'the identity is false: lhs - rhs = {symbolic(e.simplify_full(), problem)}'
    n = re.search(r'!!\[(.*?)\]', goal.text).group(1).count(';') + 1
    proof = tactic({2: 'rw [Matrix.det_fin_two_of]\nring', 3: 'norm_num [Matrix.det_fin_three]'}[n])

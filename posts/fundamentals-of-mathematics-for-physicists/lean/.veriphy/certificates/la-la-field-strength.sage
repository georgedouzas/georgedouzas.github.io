# name: la-la-field-strength
# technique: matrix computation read from the goal (matrix_from_goal recipe, 4×4 and trace of a product), with the
#   exercise's Sage solution as oracle
# goal: F^{μν} = η F η and F_{μν} F^{μν} = tr(F (F^{..})ᵀ) = 2 (B² - E²) for the antisymmetric F with F₀₁ = -E, F₁₂ = B
# description: the oracle reruns the exercise; Sage evaluates lhs - rhs from the goal; Lean proves the matrix identity
#   entrywise, and the invariant by expanding the 4×4 trace and the products (simp) and ring1.
# oracle (sage/exercises/exr-la-field-strength.sage)
E1, B3 = var('E1 B3')
eta = diagonal_matrix(SR, [1, -1, -1, -1])
F = matrix(SR, [[0, -E1, 0, 0], [E1, 0, B3, 0], [0, -B3, 0, 0], [0, 0, 0, 0]])
Fup = eta.inverse() * F * eta.inverse()
assert F.transpose() == -F and bool(sum(F[m, n] * Fup[m, n] for m in range(4) for n in range(4)) == 2 * (B3 ^ 2 - E1 ^ 2))
print('oracle: F^{mu nu} =', Fup.rows())
# recipe core
e = goal.expr
assert e is not None, 'the goal has no symbolic form'
if hasattr(e, 'nrows'):
    assert e.simplify_full() == 0, f'the identity is false: lhs - rhs = {matrix_term(e.simplify_full(), problem)}'
    proof = tactic('ext i j\nfin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply] <;> ring1')
else:
    assert bool(e.simplify_full() == 0), f'the identity is false: lhs - rhs = {symbolic(e.simplify_full(), problem)}'
    proof = tactic('simp [Matrix.trace, Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply, Matrix.vecMul, dotProduct]\nring1')

# name: set-matrix-trig
# technique: matrix computation read from the goal, checked entrywise, modulo cos² + sin² = 1 (matrix_from_goal recipe)
# goal: products and transposes of the rotation matrices R_z(α), R_x(β) with cos and sin entries: R_z(α) R_z(-α) = 1,
#   R_z(-α) = R_z(α)ᵀ, R_z(α)ᵀ R_z(α) = 1, R_z(α) R_z(β) = R_z(α + β) (addition formulas expanded), R_z R_x and R_x R_z
# description: Sage evaluates lhs - rhs from the goal and reduces it modulo cos² + sin² = 1; Lean proves each entry by
#   simp and linear_combination of that lemma.
SIMP = ['Matrix.mul_apply', 'Fin.sum_univ_succ', 'Matrix.transpose_apply']
RELATIONS = ['Real.cos_sq_add_sin_sq']
print('Sage: lhs - rhs =', goal.expr)
proof = matrix_from_goal_proof(goal, problem, simp=SIMP, relations=RELATIONS)

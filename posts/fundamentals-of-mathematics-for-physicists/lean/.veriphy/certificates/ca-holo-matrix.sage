# name: ca-holo-matrix
# technique: matrix computation read from the goal, checked entrywise, modulo side relations (matrix_from_goal recipe)
# goal: identities, determinants, products and transposes of 2×2 real Jacobians (2x, -2y; 2y, 2x), diag(1, -1),
#   eˣ (cos y, -sin y; sin y, cos y)
# description: Sage evaluates lhs - rhs from the goal and reduces it modulo cos² + sin² = 1; Lean proves each entry
#   (or the closed 2×2 determinant formula) by linear_combination of that lemma.
SIMP = []
RELATIONS = ['Real.cos_sq_add_sin_sq']
proof = matrix_from_goal_proof(goal, problem, simp=SIMP, relations=RELATIONS)

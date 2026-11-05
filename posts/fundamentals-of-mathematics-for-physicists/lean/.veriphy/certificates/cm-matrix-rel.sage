# name: cm-matrix-rel
# technique: matrix computation read from the goal, checked entrywise, modulo side relations (matrix_from_goal recipe)
# goal: identities of rotation, flow and mode matrices and vectors (cos, sin, cosh, sinh, √ entries): products, transposes,
#   determinants, eigenvector equations K a = λ M a
# description: Sage evaluates lhs - rhs from the goal and reduces it modulo cos² + sin² = 1, cosh² - sinh² = 1 and
#   (√a)² = a; Lean proves each entry by linear_combination of those lemmas.
SIMP = ['Matrix.mulVec', 'dotProduct', 'Matrix.diagonal_apply']
RELATIONS = ['Real.cos_sq_add_sin_sq', 'Real.cosh_sq_sub_sinh_sq', 'Real.sq_sqrt']
proof = matrix_from_goal_proof(goal, problem, simp=SIMP, relations=RELATIONS)

# name: fa-matrix
# technique: matrix computation read from the goal, checked entrywise, modulo side relations (matrix_from_goal recipe)
# goal: identities of the spin matrices and Wigner matrices (cos θ, sin θ, √2 entries): products, transposes, det, commutators
# description: Sage evaluates lhs - rhs from the goal and reduces it modulo cos² + sin² = 1 and (√2)² = 2; Lean proves
#   each entry by linear_combination of those lemmas.
SIMP = []
RELATIONS = ['Real.cos_sq_add_sin_sq', 'Real.sq_sqrt']
proof = matrix_from_goal_proof(goal, problem, simp=SIMP, relations=RELATIONS)

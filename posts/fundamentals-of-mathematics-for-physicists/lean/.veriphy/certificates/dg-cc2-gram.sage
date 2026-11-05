# name: dg-cc2-gram
# technique: matrix computation read from the goal, checked entrywise, modulo side relations
# goal: det (M G M^T) = (det M)^2 det G for the 2 x 2 change-of-basis matrix M and the symmetric Gram matrix G
# description: Sage evaluates lhs - rhs from the goal itself; Lean expands both determinants and closes by ring.
SIMP = []
RELATIONS = []
proof = matrix_from_goal_proof(goal, problem, simp=SIMP, relations=RELATIONS)

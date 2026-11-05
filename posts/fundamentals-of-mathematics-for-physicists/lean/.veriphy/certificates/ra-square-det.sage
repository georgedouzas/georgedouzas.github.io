# name: ra-square-det
# technique: matrix computation read from the goal, checked entrywise, modulo side relations
# goal: det (2x, -2y; 2y, 2x) = 4(x² + y²)
# description: Sage evaluates lhs - rhs from the goal (modulo cos² + sin² = 1); Lean expands det or checks entries.
SIMP = []  # EDIT: extra simp lemmas, for example definitions of named matrices
RELATIONS = []  # EDIT: side relations the entries need, as ['Real.cosh_sq_sub_sinh_sq'], ['Real.sq_sqrt']
proof = matrix_from_goal_proof(goal, problem, simp=SIMP, relations=RELATIONS)

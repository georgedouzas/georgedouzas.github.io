# name: cm-matrix-euler
# technique: matrix computation read from the goal, checked entrywise, modulo side relations (matrix_from_goal recipe),
#   with the chapter's named rotation matrices unfolded
# goal: products, transposes and determinants of the rotations Rz α, Rx α (cos α, sin α entries) of the chapter
# description: Sage evaluates lhs - rhs from the goal (the named matrices read through their literal definitions) and
#   reduces it modulo cos² + sin² = 1; Lean unfolds the definitions and proves each entry (or the expanded
#   determinant) by linear_combination of Real.cos_sq_add_sin_sq.
SIMP = ['KaluzaKlein.ClassicalMechanics.ConfigurationSpace.Rz', 'KaluzaKlein.ClassicalMechanics.ConfigurationSpace.Rx',
        'Real.cos_neg', 'Real.sin_neg']
RELATIONS = ['Real.cos_sq_add_sin_sq']
proof = matrix_from_goal_proof(goal, problem, simp=SIMP, relations=RELATIONS)

# name: dg-vf1-matrix
# technique: matrix computation read from the goal, checked entrywise, modulo side relations (matrix_from_goal recipe)
# goal: an identity of explicit 2×2 and 3×3 matrices and vectors of the differential-geometry chapter: the rotation
#   R(t) applied to a point of the sphere, the group law of plane rotations (after the addition formulas), R(0) = 1,
#   values of explicit component vectors
# description: Sage evaluates lhs - rhs from the goal itself; Lean proves it entry by entry (simp, ring1).
SIMP = ['KaluzaKlein.Geometry.rotation']
RELATIONS = []
proof = matrix_from_goal_proof(goal, problem, simp=SIMP, relations=RELATIONS)

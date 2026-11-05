# name: dg-vf2-matrix
# technique: matrix computation read from the goal, checked entrywise, modulo side relations (matrix_from_goal
#   recipe; SIMP extended with the dot product, mulVec and vecMul for pairings of covectors with vectors, RELATIONS
#   set to the Pythagorean identity for Jacobians of polar and spherical charts)
# goal: an identity, determinant, trace or pairing of explicit matrices or vectors (!![…], ![…], *, ᵀ, *ᵥ, ᵥ*, ⬝ᵥ,
#   det, trace) with numeric or trigonometric entries
# description: Sage evaluates lhs - rhs from the goal itself (cos² + sin² reduced); Lean proves it entry by entry
#   (simp with the product and pairing lemmas, ring1 or linear_combination of Real.cos_sq_add_sin_sq) or by expanding
#   det or trace.
SIMP = ['dotProduct', 'Matrix.mulVec', 'Matrix.vecMul', 'Matrix.vecMulVec_apply', 'Matrix.smul_apply', 'Matrix.one_apply']
RELATIONS = ['Real.cos_sq_add_sin_sq', 'Real.sin_sq_add_cos_sq']
proof = matrix_from_goal_proof(goal, problem, simp=SIMP, relations=RELATIONS)

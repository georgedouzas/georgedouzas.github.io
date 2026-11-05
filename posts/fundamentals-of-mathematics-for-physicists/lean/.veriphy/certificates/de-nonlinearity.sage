# name: de-nonlinearity
# technique: a failure of map_zero, map_smul or map_add at points Sage finds (nonlinearity recipe)
# goal: ¬ ∃ f : ℝ →ₗ[ℝ] ℝ, ∀ p, f p = p ^ 2 (the quadratic part of a jet expression on a jet line)
# description: Sage tests the linear laws on p ↦ p²; Lean specialises the failing law of f at the points and computes.
MAP = lambda x: x ** 2
ARITY = 1
proof = tactic(nonlinearity_proof(MAP, ARITY))

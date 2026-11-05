# name: ca-holo-chain-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.ComplexDerivative.chain
# description: the chain rule at f = g = id (z = 0) and f = w + 1, g = 2w (z = 1, i); each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['(fun w : ℂ => w)', '(fun w : ℂ => w)', '(1 : ℂ)', '(1 : ℂ)', '(0 : ℂ)', "(hasDerivAt_id' (0 : ℂ))", "(hasDerivAt_id' (0 : ℂ))"], "have h := HasDerivAt.comp (h₂ := fun w : ℂ => w) (h := fun w : ℂ => w) (0 : ℂ) (hasDerivAt_id' (0 : ℂ)) (hasDerivAt_id' (0 : ℂ))\nexact h", True),
    (['(fun w : ℂ => w + 1)', '(fun w : ℂ => 2 * w)', '(1 : ℂ)', '(2 * 1 : ℂ)', '(1 : ℂ)', "((hasDerivAt_id' (1 : ℂ)).add_const 1)", "((hasDerivAt_id' ((1 : ℂ) + 1)).const_mul 2)"], "have h := HasDerivAt.comp (h₂ := fun w : ℂ => 2 * w) (h := fun w : ℂ => w + 1) (1 : ℂ) ((hasDerivAt_id' ((1 : ℂ) + 1)).const_mul 2) ((hasDerivAt_id' (1 : ℂ)).add_const 1)\nexact h", False),
    (['(fun w : ℂ => w + 1)', '(fun w : ℂ => 2 * w)', '(1 : ℂ)', '(2 * 1 : ℂ)', 'Complex.I', "((hasDerivAt_id' Complex.I).add_const 1)", "((hasDerivAt_id' (Complex.I + 1)).const_mul 2)"], "have h := HasDerivAt.comp (h₂ := fun w : ℂ => 2 * w) (h := fun w : ℂ => w + 1) Complex.I ((hasDerivAt_id' (Complex.I + 1)).const_mul 2) ((hasDerivAt_id' Complex.I).add_const 1)\nexact h", False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]

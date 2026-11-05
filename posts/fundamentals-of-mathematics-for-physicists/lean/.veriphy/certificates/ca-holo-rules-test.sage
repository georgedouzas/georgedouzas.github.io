# name: ca-holo-rules-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.ComplexDerivative.rules
# description: sum, product, quotient rules at f = id, g = 2 and f = g = id; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['(fun w : ℂ => w)', '(fun _ : ℂ => (2 : ℂ))', '1', '0', '0', "(hasDerivAt_id' 0)", '(hasDerivAt_const 0 (2 : ℂ))'], "exact ⟨(hasDerivAt_id' _).add (hasDerivAt_const _ _), (hasDerivAt_id' _).mul (hasDerivAt_const _ _), fun h => (hasDerivAt_id' _).div (hasDerivAt_const _ _) h⟩", True),
    (['(fun w : ℂ => w)', '(fun _ : ℂ => (2 : ℂ))', '1', '0', '1', "(hasDerivAt_id' 1)", '(hasDerivAt_const 1 (2 : ℂ))'], "exact ⟨(hasDerivAt_id' _).add (hasDerivAt_const _ _), (hasDerivAt_id' _).mul (hasDerivAt_const _ _), fun h => (hasDerivAt_id' _).div (hasDerivAt_const _ _) h⟩", False),
    (['(fun w : ℂ => w)', '(fun w : ℂ => w)', '1', '1', '(1 + Complex.I)', "(hasDerivAt_id' (1 + Complex.I))", "(hasDerivAt_id' (1 + Complex.I))"], "exact ⟨(hasDerivAt_id' _).add (hasDerivAt_id' _), (hasDerivAt_id' _).mul (hasDerivAt_id' _), fun h => (hasDerivAt_id' _).div (hasDerivAt_id' _) h⟩", False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]

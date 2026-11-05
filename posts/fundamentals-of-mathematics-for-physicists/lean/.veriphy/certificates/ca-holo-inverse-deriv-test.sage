# name: ca-holo-inverse-deriv-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.HolomorphicWorked.inverse_deriv
# description: the inverse rule for f = g = id at 0, 1, i; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['(fun w : ℂ => w)', '(fun w : ℂ => w)', '1', '0', "(continuousAt_id' _)", "(hasDerivAt_id' _)", 'one_ne_zero', '(Filter.Eventually.of_forall fun _ => rfl)'], "exact HasDerivAt.of_local_left_inverse (continuousAt_id' _) (hasDerivAt_id' _) one_ne_zero (Filter.Eventually.of_forall fun _ => rfl)", True),
    (['(fun w : ℂ => w)', '(fun w : ℂ => w)', '1', '1', "(continuousAt_id' _)", "(hasDerivAt_id' _)", 'one_ne_zero', '(Filter.Eventually.of_forall fun _ => rfl)'], "exact HasDerivAt.of_local_left_inverse (continuousAt_id' _) (hasDerivAt_id' _) one_ne_zero (Filter.Eventually.of_forall fun _ => rfl)", False),
    (['(fun w : ℂ => w)', '(fun w : ℂ => w)', '1', 'Complex.I', "(continuousAt_id' _)", "(hasDerivAt_id' _)", 'one_ne_zero', '(Filter.Eventually.of_forall fun _ => rfl)'], "exact HasDerivAt.of_local_left_inverse (continuousAt_id' _) (hasDerivAt_id' _) one_ne_zero (Filter.Eventually.of_forall fun _ => rfl)", False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]

# name: ca-holo-angle-mult-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.ConformalMaps.angle_multiplied
# description: e^{inβ} = e^{in(β - α)} e^{inα} at sampled n, α, β; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
PROOF = 'rw [← Complex.exp_add]\npush_cast\nring_nf'
cases = default_cases(universal, Sampler(SEED))
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in cases]

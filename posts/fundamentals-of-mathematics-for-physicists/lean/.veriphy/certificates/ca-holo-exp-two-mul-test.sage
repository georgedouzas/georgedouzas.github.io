# name: ca-holo-exp-two-mul-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.HolomorphicWorked.exp_two_mul
# description: e^{2x} = (e^x)^2 at sampled x; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
PROOF = 'rw [sq, ← Real.exp_add]\nring_nf'
cases = default_cases(universal, Sampler(SEED))
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in cases]

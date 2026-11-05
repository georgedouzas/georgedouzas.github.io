# name: ca-holo-wirtinger-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.CauchyRiemann.wirtinger_bar_zero_iff
# description: ∂_z̄ f = 0 iff Cauchy-Riemann, at sampled partials; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
PROOF = 'norm_num [Complex.ext_iff]'
cases = default_cases(universal, Sampler(SEED))
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in cases]

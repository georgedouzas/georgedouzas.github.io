# name: ca-holo-exp-circle-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.ConformalMaps.exp_vertical_circle
# description: |e^{x0 + iy}| = e^{x0} at sampled x0, y; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
PROOF = 'rw [Complex.norm_exp]\nsimp'
cases = default_cases(universal, Sampler(SEED))
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in cases]

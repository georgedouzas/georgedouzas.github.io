# name: ca-holo-equidistant-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.ConformalMaps.real_axis_equidistant
# description: |x - i|^2 = |x + i|^2 = x^2 + 1 at sampled x; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
PROOF = 'constructor <;> simp [Complex.normSq_apply] <;> ring'
cases = default_cases(universal, Sampler(SEED))
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in cases]

# name: ca-holo-normsq-cr-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.HolomorphicWorked.normSq_cr_iff
# description: the Cauchy-Riemann equations of |z|^2 hold only at 0; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
PROOF = 'constructor <;> rintro ⟨h1, h2⟩ <;> exact ⟨by linarith, by linarith⟩'
cases = default_cases(universal, Sampler(SEED))
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in cases]

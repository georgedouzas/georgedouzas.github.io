# name: ca-holo-column-length-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.HolomorphicWorked.sq_jac_column_length
# description: (2x)^2 + (2y)^2 = (2|z|)^2 at sampled x, y; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
PROOF = 'rw [mul_pow (2 : ℝ) ‖_‖, ← Complex.normSq_eq_norm_sq, Complex.normSq_apply]\nsimp\ntry ring'
cases = default_cases(universal, Sampler(SEED))
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in cases]

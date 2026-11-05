# name: ca-holo-sq-jac-norm-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.CauchyRiemann.sq_jac_det_norm
# description: 4(x^2 + y^2) = |2z|^2 at sampled x, y; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
PROOF = 'rw [← Complex.normSq_eq_norm_sq, Complex.normSq_apply]\nnorm_num\ntry ring'
cases = default_cases(universal, Sampler(SEED))
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in cases]

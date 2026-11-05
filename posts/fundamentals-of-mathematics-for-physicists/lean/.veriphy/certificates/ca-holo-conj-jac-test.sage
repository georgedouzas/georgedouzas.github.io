# name: ca-holo-conj-jac-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.CauchyRiemann.conj_jac_not_rotation
# description: diag(1, -1) is no rotation-scaling (a, -b; b, a); each instance proved by computation or mathlib, independently of the claim.
SEED = 0
PROOF = "intro h\nhave h1 := congrFun (congrFun h 0) 0\nhave h2 := congrFun (congrFun h 1) 1\nsimp only [Matrix.of_apply, Matrix.cons_val', Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons,\n  Matrix.empty_val', Matrix.cons_val_fin_one] at h1 h2\nlinarith"
cases = default_cases(universal, Sampler(SEED))
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in cases]

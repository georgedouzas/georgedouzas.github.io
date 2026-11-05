# name: ca-holo-exp-norm-sq-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.HolomorphicWorked.exp_norm_sq
# description: |e^{x+iy}|^2 = e^{2x} at sampled x, y; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
PROOF = 'rw [Complex.norm_exp]\nsimp only [Complex.add_re, Complex.ofReal_re, Complex.mul_re, Complex.I_re, mul_zero, Complex.ofReal_im, Complex.I_im, mul_one, sub_self, add_zero]\nrw [sq, ← Real.exp_add]\nring_nf'
cases = default_cases(universal, Sampler(SEED))
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in cases]

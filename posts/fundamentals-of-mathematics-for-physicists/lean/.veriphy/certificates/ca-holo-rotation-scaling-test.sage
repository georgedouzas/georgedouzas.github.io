# name: ca-holo-rotation-scaling-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.ConformalMaps.rotation_scaling
# description: a h = |a| e^{i arg a} h at (0, 0), (1 + i, 2), (i, i); each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['0', '0'], 'rw [← mul_assoc, Complex.norm_mul_exp_arg_mul_I]', True),
    (['(1 + Complex.I)', '2'], 'rw [← mul_assoc, Complex.norm_mul_exp_arg_mul_I]', False),
    (['Complex.I', 'Complex.I'], 'rw [← mul_assoc, Complex.norm_mul_exp_arg_mul_I]', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]

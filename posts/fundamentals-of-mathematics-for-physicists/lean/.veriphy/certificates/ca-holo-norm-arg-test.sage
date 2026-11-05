# name: ca-holo-norm-arg-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.ConformalMaps.norm_arg_diag
# description: |a(1 + i)| = a√2 and arg = π/4 at a = 1, 2, 1/2; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['1', 'one_pos'], 'exact KaluzaKlein.ComplexAnalysis.ConformalMaps.norm_arg_diag_aux 1 one_pos', True),
    (['2', 'two_pos'], 'exact KaluzaKlein.ComplexAnalysis.ConformalMaps.norm_arg_diag_aux 2 two_pos', False),
    (['(1 / 2)', '(by norm_num)'], 'exact KaluzaKlein.ComplexAnalysis.ConformalMaps.norm_arg_diag_aux (1 / 2) (by norm_num)', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]

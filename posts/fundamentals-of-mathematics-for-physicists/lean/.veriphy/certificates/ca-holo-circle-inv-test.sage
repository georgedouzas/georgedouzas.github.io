# name: ca-holo-circle-inv-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.HolomorphicWorked.circleIntegral_sub_inv
# description: ∮ dz/(z - a) is 2πi inside and 0 outside, for a = 0, 2, i; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['0', '0', '1', 'one_pos'], 'exact KaluzaKlein.ComplexAnalysis.HolomorphicWorked.circleIntegral_sub_inv_aux 0 0 1 one_pos', True),
    (['2', '0', '1', 'one_pos'], 'exact KaluzaKlein.ComplexAnalysis.HolomorphicWorked.circleIntegral_sub_inv_aux 2 0 1 one_pos', False),
    (['Complex.I', '0', '2', 'two_pos'], 'exact KaluzaKlein.ComplexAnalysis.HolomorphicWorked.circleIntegral_sub_inv_aux Complex.I 0 2 two_pos', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]

# name: ca-holo-winding-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.HolomorphicWorked.winding_numbers
# description: winding numbers +1, -1, -1 around 1, i, -i on circles of radius 1/2, 1/4, 9/10; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['(1 / 2)', '(by norm_num)', '(by norm_num)'], 'exact KaluzaKlein.ComplexAnalysis.HolomorphicWorked.winding_numbers_aux (1 / 2) (by norm_num) (by norm_num)', True),
    (['(1 / 4)', '(by norm_num)', '(by norm_num)'], 'exact KaluzaKlein.ComplexAnalysis.HolomorphicWorked.winding_numbers_aux (1 / 4) (by norm_num) (by norm_num)', False),
    (['(9 / 10)', '(by norm_num)', '(by norm_num)'], 'exact KaluzaKlein.ComplexAnalysis.HolomorphicWorked.winding_numbers_aux (9 / 10) (by norm_num) (by norm_num)', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]

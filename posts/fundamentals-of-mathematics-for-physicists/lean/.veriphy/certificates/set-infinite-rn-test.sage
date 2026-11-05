# name: set-infinite-rn-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Sets.infinite_rn
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['1', '(by norm_num)'], True, 'infer_instance'),
    (['2', '(by norm_num)'], False, 'infer_instance'),
    (['4', '(by norm_num)'], False, 'infer_instance'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]

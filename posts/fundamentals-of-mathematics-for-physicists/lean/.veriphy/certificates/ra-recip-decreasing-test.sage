# name: ra-recip-decreasing-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Sequences.recip_decreasing file "ra-recip-decreasing-test.sage"
# description: 0 < 1/(n+2) < 1/(n+1) at n = 0, 1, 5, 99
SEED = 0  # the instances are listed, no sampling
PROOF = 'constructor <;> norm_num'
CASES = [(['0'], True, None), (['1'], False, None), (['5'], False, None), (['99'], False, None)]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]

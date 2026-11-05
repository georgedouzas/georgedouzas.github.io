# name: ra-newton-fixed-point-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Sequences.newton_fixed_point file "ra-newton-fixed-point-test.sage"
# description: L = (L + 2/L)/2 exactly when L² = 2, checked at rational L (both sides false)
SEED = 0  # the instances are listed, no sampling
PROOF = 'norm_num'
CASES = [(['1', '(by norm_num)'], True, None), (['2', '(by norm_num)'], False, None), (['-1', '(by norm_num)'], False, None), (['3 / 2', '(by norm_num)'], False, None)]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]

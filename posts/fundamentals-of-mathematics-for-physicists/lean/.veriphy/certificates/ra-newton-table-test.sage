# name: ra-newton-table-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Sequences.newton_table file "ra-newton-table-test.sage"
# description: the iterates 2, 3/2, 17/12, 577/408 computed exactly
SEED = 0  # the instances are listed, no sampling
PROOF = 'norm_num [KaluzaKlein.RealAnalysis.Sequences.newton]'
CASES = [(['0'], True, None), (['1'], False, None), (['2'], False, None), (['3'], False, None)]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]

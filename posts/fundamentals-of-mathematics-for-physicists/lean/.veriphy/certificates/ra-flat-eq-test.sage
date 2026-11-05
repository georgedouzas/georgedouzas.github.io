# name: ra-flat-eq-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Continuity.flat_eq file "ra-flat-eq-test.sage"
# description: f(x) = e^(-1/x²) off 0, unfolded exactly
SEED = 0  # the instances are listed, no sampling
PROOF = 'unfold KaluzaKlein.RealAnalysis.Continuity.flat\nrw [expNegInvGlue, if_neg (by norm_num)]\nnorm_num'
CASES = [(['1', '(by norm_num)'], True, None), (['-2', '(by norm_num)'], False, None), (['1 / 3', '(by norm_num)'], False, None)]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]

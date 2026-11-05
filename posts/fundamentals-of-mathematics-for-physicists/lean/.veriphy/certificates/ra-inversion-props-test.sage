# name: ra-inversion-props-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Calculus.inversion_props file "ra-inversion-props-test.sage"
# description: the inversion is an involution with det J = -1/(u² + v²)², exactly at sample points
SEED = 0  # the instances are listed, no sampling
PROOF = 'constructor\n· norm_num\n· rw [Matrix.det_fin_two_of]; norm_num'
CASES = [(['1', '0', '(by norm_num)'], True, None), (['1', '2', '(by norm_num)'], False, None), (['-3', '1 / 2', '(by norm_num)'], False, None)]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]

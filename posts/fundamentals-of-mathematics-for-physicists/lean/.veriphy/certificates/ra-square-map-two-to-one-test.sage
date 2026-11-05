# name: ra-square-map-two-to-one-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Calculus.square_map_two_to_one file "ra-square-map-two-to-one-test.sage"
# description: F(x, y) = F(-x, -y) at sample points, exactly
SEED = 0  # the instances are listed, no sampling
PROOF = 'norm_num'
CASES = [(['0', '0'], True, None), (['1', '2'], False, None), (['-3', '1 / 2'], False, None)]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]

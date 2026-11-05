# name: ra-deriv-pow-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Continuity.deriv_pow_rule file "ra-deriv-pow-test.sage"
# description: (x^n)' = n x^(n-1) at sample n, x
SEED = 0  # the instances are listed, no sampling
PROOF = 'simp'
CASES = [(['0', '5'], True, None), (['3', '2'], False, None), (['5', '-1'], False, None), (['2', '1 / 2'], False, None)]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]

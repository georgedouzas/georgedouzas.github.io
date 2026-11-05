# name: ra-sqrt-prime-irrational-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Sequences.sqrt_prime_irrational file "ra-sqrt-prime-irrational-test.sage"
# description: √p irrational, by norm_num's irrationality extension, at p = 2, 3, 5
SEED = 0  # the instances are listed, no sampling
PROOF = 'exact Nat.Prime.irrational_sqrt (by norm_num)'
CASES = [(['2', 'Nat.prime_two'], True, None), (['3', 'Nat.prime_three'], False, None), (['5', '(by norm_num)'], False, None)]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]

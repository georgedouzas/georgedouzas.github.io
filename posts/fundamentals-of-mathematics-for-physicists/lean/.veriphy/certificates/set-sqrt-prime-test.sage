# name: set-sqrt-prime-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Sets.sqrt_prime_irrational
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['2', '(by norm_num)'], True, 'exact Nat.prime_two.irrational_sqrt'),
    (['3', '(by norm_num)'], False, 'exact Nat.prime_three.irrational_sqrt'),
    (['5', '(by norm_num)'], False, 'exact Nat.prime_five.irrational_sqrt'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]

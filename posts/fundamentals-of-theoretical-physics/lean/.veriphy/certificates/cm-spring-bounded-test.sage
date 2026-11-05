# name: cm-spring-bounded-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback recipe)
# goal: sage_test KaluzaKlein.ClassicalMechanics.Energy.spring_bounded file "cm-spring-bounded-test.sage"
# description: k x² ≤ 2E for E = m v²/2 + k x²/2, at (m, k, x, v) = (1, 1, 0, 0) (edge), (2, 3, 1, 2), (1, 1/2, 2, 1)
SEED = 0  # the instances are listed by hand
PROOF = 'norm_num'
CASES = [(['1', '1', '0', '0', '0', '(by norm_num)', '(by norm_num)'], True),
         (['2', '3', '1', '2', '11/2', '(by norm_num)', '(by norm_num)'], False),
         (['1', '1/2', '2', '1', '3/2', '(by norm_num)', '(by norm_num)'], False)]
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in CASES]

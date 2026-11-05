# name: cm-escape-iff-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback recipe)
# goal: sage_test KaluzaKlein.ClassicalMechanics.Energy.escape_iff file "cm-escape-iff-test.sage"
# description: E ≥ 0 ↔ v² ≥ 2GM/R at (G, M, m, R, v) = (0, 1, 1, 1, 0) (edge), (1, 1, 1, 1, 1) (bound), (1, 1, 1, 1, 2) (escapes)
SEED = 0  # the instances are listed by hand
PROOF = 'norm_num'
CASES = [(['0', '1', '1', '1', '0', '(by norm_num)', '(by norm_num)'], True),
         (['1', '1', '1', '1', '1', '(by norm_num)', '(by norm_num)'], False),
         (['1', '1', '1', '1', '2', '(by norm_num)', '(by norm_num)'], False)]
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in CASES]

# name: cm-magnetic-not-potential-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback recipe)
# goal: sage_test KaluzaKlein.ClassicalMechanics.ConfigurationSpace.magnetic_not_potential file "cm-magnetic-not-potential-test.sage"
# description: (qB/2)(x ẏ - y ẋ) is no function of the position alone, at (q, B) = (1, 1) (edge), (-1, 2), (1/2, 3)
SEED = 0  # the instances are listed by hand
PROOF = 'rintro ⟨V, hV⟩\nhave h0 := hV 1 0 0 0\nhave h1 := hV 1 0 0 1\nnorm_num at h0 h1\nlinarith'
CASES = [(['1', '1', '(by norm_num)'], True), (['-1', '2', '(by norm_num)'], False), (['1/2', '3', '(by norm_num)'], False)]
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in CASES]

# name: cm-kepler-U-second-pos-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback recipe)
# goal: sage_test KaluzaKlein.ClassicalMechanics.CentralForces.kepler_U_second_pos file "cm-kepler-U-second-pos-test.sage"
# description: U''(r₀) = m³k⁴/L⁶ > 0 at (L, m, k) = (1, 1, 1) (edge), (-2, 3, 1/2), (1/3, 1/2, 5)
SEED = 0  # the instances are listed by hand
PROOF = 'positivity'
CASES = [(['1', '1', '1', '(by norm_num)', '(by norm_num)', '(by norm_num)'], True),
         (['-2', '3', '1/2', '(by norm_num)', '(by norm_num)', '(by norm_num)'], False),
         (['1/3', '1/2', '5', '(by norm_num)', '(by norm_num)', '(by norm_num)'], False)]
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in CASES]

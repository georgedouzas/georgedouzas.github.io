# name: cm-conic-type-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback recipe)
# goal: sage_test KaluzaKlein.ClassicalMechanics.Kepler.conic_type_iff file "cm-conic-type-test.sage"
# description: the sign of E = k(e² - 1)/(2p) against e at (k, p, e) = (1, 1, 0) (edge, circle), (1, 1, 1) (parabola), (1, 2, 2) (hyperbola)
SEED = 0  # the instances are listed by hand
PROOF = 'norm_num'
H = ['(by norm_num)'] * 4
CASES = [(['1', '1', '0', '-1/2'] + H, True), (['1', '1', '1', '0'] + H, False), (['1', '2', '2', '3/4'] + H, False)]
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in CASES]

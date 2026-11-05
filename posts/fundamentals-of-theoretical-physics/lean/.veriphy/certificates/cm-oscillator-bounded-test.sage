# name: cm-oscillator-bounded-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback recipe)
# goal: sage_test KaluzaKlein.ClassicalMechanics.CentralForces.oscillator_bounded file "cm-oscillator-bounded-test.sage"
# description: k r² ≤ 2E at (m, k, L, r, ṙ) = (1, 1, 0, 1, 0) (edge, E = 1/2), (1, 2, 1, 1, 1) (E = 2), (2, 1, 2, 2, 0) (E = 9/4)
SEED = 0  # the instances are listed by hand
PROOF = 'norm_num'
H = ['(by norm_num)'] * 2
CASES = [(['1', '1', '1/2', '0', '1', '0'] + H, True), (['1', '2', '2', '1', '1', '1'] + H, False), (['2', '1', '9/4', '2', '2', '0'] + H, False)]
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in CASES]

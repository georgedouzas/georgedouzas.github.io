# name: cm-energy-above-minimum-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback recipe)
# goal: sage_test KaluzaKlein.ClassicalMechanics.CentralForces.energy_above_minimum_iff file "cm-energy-above-minimum-test.sage"
# description: E > -mk²/(2L²) ↔ m²k² + 2mEL² > 0 at (m, k, E, L) = (1, 1, -1/2, 1) (edge, the minimum), (1, 1, 0, 1), (2, 1, -1, 1)
SEED = 0  # the instances are listed by hand
PROOF = 'norm_num'
H = ['(by norm_num)'] * 2
CASES = [(['1', '1', '-1/2', '1'] + H, True), (['1', '1', '0', '1'] + H, False), (['2', '1', '-1', '1'] + H, False)]
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in CASES]

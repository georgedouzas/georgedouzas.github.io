# name: cm-two-turning-points-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback recipe)
# goal: sage_test KaluzaKlein.ClassicalMechanics.CentralForces.two_turning_points_iff file "cm-two-turning-points-test.sage"
# description: u₋ > 0 ↔ E < 0 at (m, k, L, E, s) = (1, 1, 1, 0, 1) (edge, parabola), (1, 1, 1, -3/8, 1/2), (1, 1, 1, 3/2, 2)
SEED = 0  # the instances are listed by hand
PROOF = 'norm_num'
H = ['(by norm_num)'] * 5
CASES = [(['1', '1', '0', '1', '1'] + H, True), (['1', '1', '-3/8', '1', '1/2'] + H, False), (['1', '1', '3/2', '1', '2'] + H, False)]
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in CASES]

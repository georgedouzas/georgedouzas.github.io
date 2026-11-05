# name: cm-powerlaw-stable-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback recipe)
# goal: sage_test KaluzaKlein.ClassicalMechanics.CentralForces.powerlaw_stable_iff file "cm-powerlaw-stable-test.sage"
# description: (3 + α) L²/(m r₀⁴) > 0 ↔ α > -3 at α = -3 (edge, marginal), -2 (gravity), 1 (oscillator), -4 (unstable)
SEED = 0  # the instances are listed by hand
PROOF = 'norm_num'
H = ['(by norm_num)'] * 3
CASES = [(['1', '1', '1', '-3'] + H, True), (['1', '1', '1', '-2'] + H, False), (['1', '1', '1', '1'] + H, False),
         (['2', '1', '1', '-4'] + H, False)]
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in CASES]

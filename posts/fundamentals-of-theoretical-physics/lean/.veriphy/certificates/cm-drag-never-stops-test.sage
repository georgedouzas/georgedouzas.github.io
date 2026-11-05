# name: cm-drag-never-stops-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback recipe)
# goal: sage_test KaluzaKlein.ClassicalMechanics.Momentum.drag_never_stops file "cm-drag-never-stops-test.sage"
# description: v₀ e^{-γ t/m} > 0 at (m, γ, v₀, t) = (1, 1, 1, 0) (edge), (2, 1/2, 3, 5), (1, -1, 1/10, -2)
SEED = 0  # the instances are listed by hand
PROOF = 'positivity'
CASES = [(['1', '1', '1', '(by norm_num)', '0'], True), (['2', '1/2', '3', '(by norm_num)', '5'], False),
         (['1', '-1', '1/10', '(by norm_num)', '-2'], False)]
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in CASES]
